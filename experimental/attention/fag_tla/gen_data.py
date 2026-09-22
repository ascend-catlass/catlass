# TLA-style FAG (Flash Attention Gradient) backward pass data generation
import os
import torch
import torch_npu
import numpy as np
import sys
import random
from einops import rearrange

if len(sys.argv) >= 8:
    torch.npu.set_device(int(sys.argv[7]))
np.random.seed(3)
torch.manual_seed(3)
# random.seed(3)

torch.set_printoptions(threshold=float('inf'))

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def save_file(filename, save_tensor):
    file_path = os.path.join(WORKSPACE, "data", filename)
    with open(file_path, "wb") as f:
        # 直接写出原始内存字节，不做任何类型转换
        f.write(save_tensor.view(torch.uint8).cpu().detach().numpy().tobytes())

def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum

def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def tforward(q, k, v, drop_mask, atten_mask, pse, scale, keep_prob):
    print('tforward running')
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)) + pse).mul(scale)
    if atten_mask is None or len(atten_mask.shape) == 0:
        qk = qk
    else:
        qk = qk + atten_mask * (-40000.0)  # -10000
    softmax_res, x_max, x_sum = tsoftmax(qk)
    if atten_mask is not None:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0 # 要斟酌一下，部分场景存在计算异常精度问题
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    y = torch.matmul(drop_res, v)
    return y, softmax_res


def tbackward(dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob):
    print('tbackward running')
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / (keep_prob))
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res))
    dq = torch.matmul(softmax_grad_res, k)
    dq = dp * scale # post
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)

    # 保存数据
    print(f"ds dtype={softmax_grad_res.dtype}")
    softmax_grad_res.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "ds.bin"))  # BNS1S2

    return dq, dk, dv

def get_cu_seqlens(seqlens_list):
    if seqlens_list is None:
        return None
    cu = torch.zeros(len(seqlens_list) + 1, dtype = torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu

def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, batch: int):
    if max_q_seqlen <=0 or max_kv_seqlen <=0:
        return None, None
    q_seqlen_list = []
    kv_seqlen_list = []
    for i in range(batch):
        q_seq = random.randint(1, max_q_seqlen)
        kv_seq = random.randint(1, max_kv_seqlen)
        q_seqlen_list.append(q_seq)
        kv_seqlen_list.append(kv_seq)

    return q_seqlen_list, kv_seqlen_list

def gen_data(batch, nheads, nheads_k, seqlen_q, seqlen_kv, headdim, dtype, layout, isDtm, max_q_seqlen, max_kv_seqlen):
    pttype = torch.float16 if dtype=="half" else torch.bfloat16
    # if os.path.exists(os.path.join(WORKSPACE, "data", "q.bin")) and np.fromfile(os.path.join(WORKSPACE, "data", "q.bin"), dtype=np.float16).size == np.prod(batch, seqlen_q, nheads_k, headdim):
    if os.path.exists(os.path.join(WORKSPACE, "data", "q.bin")) and False:
        print("golden is exits!")
        q = torch.from_numpy(np.fromfile(os.path.join(WORKSPACE, "data", "q.bin"), dtype=np.float16).reshape(batch, seqlen_q, nheads_k, headdim))
        k = torch.from_numpy(np.fromfile(os.path.join(WORKSPACE, "data", "k.bin"), dtype=np.float16).reshape(batch, seqlen_kv, nheads_k, headdim))
        v = torch.from_numpy(np.fromfile(os.path.join(WORKSPACE, "data", "v.bin"), dtype=np.float16).reshape(batch, seqlen_kv, nheads_k, headdim))
        dout = torch.from_numpy(np.fromfile(os.path.join(WORKSPACE, "data", "dout.bin"), dtype=np.float16).reshape(batch, seqlen_q, nheads_k, headdim))
        atten_mask = torch.from_numpy(np.fromfile(os.path.join(WORKSPACE, "data", "atten_mask.bin"), dtype=np.bool_).reshape(2048, 2048))
        q = q.npu()
        k = k.npu()
        v = v.npu()
        dout = dout.npu()
        atten_mask_npu = atten_mask.npu()

        scale = 1 / (headdim ** 0.5)
        keep_prob = 1.0
        pre_tocken = 0
        next_tocken = 0

        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        torch.npu.synchronize()
        npu_rst = torch_npu.npu_fusion_attention(
                q, k, v, nheads,
                pse=None,
                padding_mask=None,
                # atten_mask=atten_mask_npu,
                atten_mask=None,
                scale=scale,
                keep_prob=keep_prob,
                input_layout="BSND",
                # actual_seq_qlen=tuple(cu_seq_len_list),
                # actual_seq_kvlen=tuple(cu_seq_kvlen_list),
                pre_tockens=pre_tocken,
                next_tockens=next_tocken,
                inner_precise=0,
                # sparse_mode=2,
                sparse_mode=0,
                prefix=None)
        out_npu = npu_rst[0]
        x_max_npu = npu_rst[1]
        x_sum_npu = npu_rst[2]
        torch.npu.synchronize()
        out_npu.backward(dout)
        dq_golden_npu = q.grad
        dk_golden_npu = k.grad
        dv_golden_npu = v.grad
        torch.npu.synchronize()

        print("soft_max_max shape ", x_max_npu.shape, x_max_npu.dtype)
        print("soft_max_sum shape ", x_sum_npu.shape, x_sum_npu.dtype)
        print("attention in shape ", out_npu.shape, out_npu.dtype)

        print("dq_golden shape ", dq_golden_npu.shape, dq_golden_npu.dtype)
        print("dk_golden shape ", dk_golden_npu.shape, dk_golden_npu.dtype)
        print("dv_golden shape ", dv_golden_npu.shape, dv_golden_npu.dtype)

        return
    else:
        if isDtm == 1:
            print("torch using deterministic now")
            torch.use_deterministic_algorithms(True)
        g = nheads / nheads_k
        scale = 1 / (headdim ** 0.5)
        pre_tocken = 0
        next_tocken = 0
        keep_prob = 1.0
        limit = 2

        actual_seq_qlen = None
        actual_seq_kvlen = None

        if layout == "BSND":
            q = limit * (torch.rand([batch, seqlen_q, nheads, headdim]) - 0.5).to(pttype)
            k = limit * (torch.rand([batch, seqlen_kv, nheads_k, headdim]) - 0.5).to(pttype)
            v = limit * (torch.rand([batch, seqlen_kv, nheads_k, headdim]) - 0.5).to(pttype)
            dout = limit * (torch.rand([batch, seqlen_q, nheads, headdim]) - 0.5).to(pttype)
        elif layout == "SBH":
            q = limit * (torch.rand([seqlen_q, batch, nheads * headdim]) - 0.5).to(pttype)
            k = limit * (torch.rand([seqlen_kv, batch, nheads_k * headdim]) - 0.5).to(pttype)
            v = limit * (torch.rand([seqlen_kv, batch, nheads_k * headdim]) - 0.5).to(pttype)
            dout = limit * (torch.rand([seqlen_q, batch, nheads * headdim]) - 0.5).to(pttype)
        elif layout == "BNSD":
            q = limit * (torch.rand([batch, nheads, seqlen_q, headdim]) - 0.5).to(pttype)
            k = limit * (torch.rand([batch, nheads_k, seqlen_kv, headdim]) - 0.5).to(pttype)
            v = limit * (torch.rand([batch, nheads_k, seqlen_kv, headdim]) - 0.5).to(pttype)
            dout = limit * (torch.rand([batch, nheads, seqlen_q, headdim]) - 0.5).to(pttype)
        elif layout == "TND":
            seqlens_list_q, seqlens_list_kv = gen_seqlen(max_q_seqlen, max_kv_seqlen, batch)
            cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
            cu_seqlens_kv = get_cu_seqlens(seqlens_list_kv)
            total_q = cu_seqlens_q[len(seqlens_list_q)]
            total_kv = cu_seqlens_kv[len(seqlens_list_kv)]
            q = limit * (torch.rand([total_q, nheads, headdim]) - 0.5).to(pttype)
            k = limit * (torch.rand([total_kv, nheads_k, headdim]) - 0.5).to(pttype)
            v = limit * (torch.rand([total_kv, nheads_k, headdim]) - 0.5).to(pttype)
            dout = limit * (torch.rand([total_q, nheads, headdim]) - 0.5).to(pttype)

            actual_seq_qlen = tuple(cu_seqlens_q[1:].cpu().numpy().tolist())
            actual_seq_kvlen = tuple(cu_seqlens_kv[1:].cpu().numpy().tolist())

        q = q.npu()
        k = k.npu()
        v = v.npu()
        dout = dout.npu()

        print("q.shape ", q.shape)
        print("k.shape ", k.shape)
        print("v.shape ", v.shape)
        print("dout.shape ", dout.shape)
        if layout == "TND":
            print("seqlens_list_q is ", seqlens_list_q)
            print("seqlens_list_kv is ", seqlens_list_kv)
            print("cu_seq_len_list is ", actual_seq_qlen)
            print("cu_seq_kvlen_list is ", actual_seq_kvlen)

        atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()

        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        torch.npu.synchronize()
        npu_rst = torch_npu.npu_fusion_attention(
                q, k, v, nheads,
                pse=None,
                padding_mask=None,
                # atten_mask=atten_mask_npu,
                atten_mask=None,
                scale=scale,
                keep_prob=keep_prob,
                input_layout=layout,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                pre_tockens=pre_tocken,
                next_tockens=next_tocken,
                inner_precise=0,
                # sparse_mode=2,
                sparse_mode=0,
                prefix=None)
        out_npu = npu_rst[0]
        x_max_npu = npu_rst[1]
        x_sum_npu = npu_rst[2]
        torch.npu.synchronize()
        out_npu.backward(dout)
        dq_golden_npu = q.grad
        dk_golden_npu = k.grad
        dv_golden_npu = v.grad
        torch.npu.synchronize()

        print("soft_max_max shape ", x_max_npu.shape, x_max_npu.dtype)
        print("soft_max_sum shape ", x_sum_npu.shape, x_sum_npu.dtype)
        print("attention in shape ", out_npu.shape, out_npu.dtype)

        print("dq_golden shape ", dq_golden_npu.shape, dq_golden_npu.dtype)
        print("dk_golden shape ", dk_golden_npu.shape, dk_golden_npu.dtype)
        print("dv_golden shape ", dv_golden_npu.shape, dv_golden_npu.dtype)

        save_file("q.bin", q.cpu().detach())
        save_file("k.bin", k.cpu().detach())
        save_file("v.bin", v.cpu().detach())
        save_file("dout.bin", dout.cpu().detach())
        save_file("out.bin", out_npu.cpu().detach())
        atten_mask_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "atten_mask.bin"))
        # x_max_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "row_max.bin"))
        # x_sum_npu.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "row_sum.bin"))

        softmax_lse = x_max_npu.cpu() + torch.log(x_sum_npu.cpu()) # bns8 or tn8
        if layout == "TND":
            np.array(actual_seq_qlen).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "q_seqlen.bin"))
            np.array(actual_seq_kvlen).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "kv_seqlen.bin"))
            np.array([total_q]).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
            np.array([total_kv]).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens.bin"))
            # softmax_lse = softmax_lse[..., 0].transpose(0, 1) # nt
            softmax_lse = softmax_lse[..., 0] # TN
        else:
            softmax_lse = softmax_lse[..., 0].transpose(1, 2) # bsn
        
        print("softmax_lse shape ", softmax_lse.shape, softmax_lse.dtype)
        softmax_lse.cpu().detach().numpy().tofile(os.path.join(WORKSPACE, "data", "softmax_lse.bin"))

        dq_golden_npu.cpu().to(torch.float).numpy().tofile(os.path.join(WORKSPACE, "data", "dq_golden.bin"))
        dk_golden_npu.cpu().to(torch.float).numpy().tofile(os.path.join(WORKSPACE, "data", "dk_golden.bin"))
        dv_golden_npu.cpu().to(torch.float).numpy().tofile(os.path.join(WORKSPACE, "data", "dv_golden.bin"))

if __name__ == '__main__':
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    batch = int(sys.argv[1])
    nheads = int(sys.argv[2])
    nheads_k = int(sys.argv[3])
    seqlen_q = int(sys.argv[4])
    seqlen_kv = int(sys.argv[5])
    headdim = int(sys.argv[6])
    dtype = sys.argv[8]
    layout = sys.argv[9]
    isDtm = int(sys.argv[10])
    max_q_seqlen = int(sys.argv[11])
    max_kv_seqlen = int(sys.argv[12])

    gen_data(batch, nheads, nheads_k, seqlen_q, seqlen_kv, headdim, dtype, layout, isDtm, max_q_seqlen, max_kv_seqlen)
