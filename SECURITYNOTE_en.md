# CATLASS Repository Security Statement

## System Security Hardening

- You can enable ASLR (Level 2) during system configuration to enhance system security and ensure system randomization is activated.
  Refer to the following method for configuration:

  ```sh
  echo 2 > /proc/sys/kernel/randomize_va_space
  ```

## User Account Recommendations

- To ensure security and minimize permissions, you are not advised to use administrator accounts such as `root`.

## File Permission Control

- You are advised to set the `umask` value of the running system to `0027` or higher on the host (including the host) and container, to ensure that the default highest permission of new folders is `750` and that of new files is `640`.
- You are advised to take security measures such as permission control on sensitive content, including personal privacy data, business assets, source files, and various files saved during operator development. For example, control the permissions of the installation directory related to the CATLASS repository and the public data files input to operators. The recommended permissions can be set with reference to [Appendix A Recommended Maximum Permission Control Value for Files (Folders) in Each Scenario](#a-recommended-maximum-permissions-for-files-and-folders).
- During operator runtime, compiled operator files may be cached in the `kernel_meta_*` folder under the running directory to accelerate subsequent operator calls. You can independently control the permissions of the generated files as needed.
- You must implement proper permission control during the installation and usage process, with reference to the file permission guidelines in [Appendix A Recommended Maximum Permission Control Value for Files (Folders) in Each Scenario](#a-recommended-maximum-permissions-for-files-and-folders).

## Build Security Statement

- When compiling and running CATLASS examples, some intermediate files will be generated. It is recommended to set appropriate permissions for these intermediate files after compilation to ensure file security.

## Runtime Security Statement

- You are advised to write an operator calling script based on the operating environment resources. If the operator calling script does not match the resource status, for example, the space used for generating input data and benchmark computing results exceeds the memory capacity limit, or the data stored locally in the script exceeds the disk space, an error may occur and the process may exit unexpectedly.
- If an operator exits abnormally during runtime, it will terminate the process and print error information. It is recommended to locate the specific cause of the error based on the error prompts, including enabling the operator synchronous execution function and checking CANN log files.
- When operators are called using TorchNPU, [TorchNPU](https://gitcode.com/Ascend/pytorch) is used. This may result in runtime errors due to version mismatch. For details, see [TorchNPU Security Statement](https://gitcode.com/Ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E).

## Sample Security Statement

- [CATLASS sample](./examples) aims to provide a minimal implementation for quickly getting started with, developing, and debugging CATLASS operators. Its core objective is to demonstrate CATLASS's core functionality using the simplest possible code, rather than providing production-level security. Compared to mature inference frameworks, the security features (such as input validation and boundary checks) in this sample are relatively limited.
- CATLASS does not recommend that you directly use the sample as the service code, and does not ensure the security of such practices. If you directly apply these sample codes to their real business scenarios and any related security issues occur, the responsibility shall be borne solely by you.

## Interface Security Statement

- CATLASS's core responsibility is to provide high-performance and functionally correct operator template implementations (located in [include](./include)). To pursue optimal NPU performance, the library implementation includes (but is not limited to) the following design trade-offs:
  - No address validation for buffer (e.g., UB/L1/L0) accesses
  - No validation of the representation range for large-to-small type conversions
  - No division-by-zero validation for division operations (e.g., Scalar `/` or Vector `Div`)
  - No validation of the validity of loop conditions passed from external sources
  - No validation to prevent overflow of arithmetic operation results
- This design is based on the principle of performance priority: NPU validation relies on scalar units, and additional security instructions will cause performance degradation, so runtime security checks are intentionally omitted.
- CATLASS only guarantees the functional correctness of operator templates under valid inputs. Any security risks arising from invalid inputs (e.g., zero/overflow values as divisors, illegal memory accesses, or unmet boundary conditions) shall be borne by the caller.

## Public Network Address Statement

- Currently, the CATLASS repository code does not involve public network addresses.

---

## Appendix

### A-Recommended maximum permissions for files and folders

| Type                                                                      | Maximum Linux Permission |
| ------------------------------------------------------------------------- | ------------------------ |
| Home directory                                                            | 750 (rwxr-x---)          |
| Program files (including scripts and libraries)                           | 550 (r-xr-x---)          |
| Program file directory                                                    | 550 (r-xr-x---)          |
| Configuration files                                                       | 640 (rw-r-----)          |
| Configuration file directory                                              | 750 (rwxr-x---)          |
| Log files (recorded or archived)                                          | 440 (r--r-----)          |
| Log files (being recorded)                                                | 640 (rw-r-----)          |
| Log file directory                                                        | 750 (rwxr-x---)          |
| Debug files                                                               | 640 (rw-r-----)          |
| Debug file directory                                                      | 750 (rwxr-x---)          |
| Temporary file directory                                                  | 750 (rwxr-x---)          |
| Maintenance and upgrade file directory                                    | 770 (rwxrwx---)          |
| Service data files                                                        | 640 (rw-r-----)          |
| Service data file directory                                               | 750 (rwxr-x---)          |
| Key components, private keys, certificates, and ciphertext file directory | 700 (rwx------)          |
| Key components, private keys, certificates, and ciphertext files          | 600 (rw-------)          |
| APIs and scripts for encryption and decryption                            | 500 (r-x------)          |
