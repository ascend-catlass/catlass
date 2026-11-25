# Read the system cycle counter

`tla.arch.get_system_cycle()` reads the issuing core's system counter as a
64-bit `tla.Int64` bit pattern. Use it in scalar code inside a Cube or Vector
region. Subtract two reads on the same physical issuer to measure elapsed
system ticks.

```python
with tla.vector():
    before = tla.arch.get_system_cycle()
    after = tla.arch.get_system_cycle()
    ticks = after - before
```

The read inserts no synchronization. The example drains earlier work before
the first read and waits for Vector retirement before the second. The
measurement includes Scalar work and synchronization overhead. The reported
units are system ticks at the target's counter frequency.

On a selected Ascend 950 device, run the example with the normal device mask
and the process-local device index.

```bash
python examples/python/tla_dsl/tla/kernel/system_counter/system_counter.py --device 0
```
