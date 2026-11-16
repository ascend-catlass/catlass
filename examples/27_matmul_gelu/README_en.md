# MatmulGelu Example Readme

## Code Organization

```text
├── 27_matmul_gelu
│   ├── CMakeLists.txt # CMake build file
│   ├── README.md
│   └── matmul_gelu.cpp # Main file
```

## Function

Performs the computation of the following functions:

Gelu:

$$
out = Gelu(a*b)

$$

Where the formula for Gelu is:

$$
Gelu(x) =0.5*x*(1 + Tanh(\sqrt {2/π}*(x + 0.044715*x^3 )))
$$

Sigmoid:

$$
Sigmoid(x)=\frac{1}{1+e^{-x}}
$$

Tanh:

$$
\begin{aligned}
Tanh(x) &= \frac{(e^x - e^{-x})}{(e^x + e^{-x})}\\
&= \frac{(e^x - e^{-x})\times e^{-x}}{(e^x + e^{-x}) \times e^{-x} }\\
&= \frac{1 - e^{-2x} }{1 + e^{-2x}}\\
&= 1 - 2\times \frac{e^{-2x}}{1 + e^{-2x}}\\
&= 1 - 2\times (1 - \frac{1}{1 + e^{-2x}})\\
&= 1 - 2\times (1 - Sigmoid(2x))
\end{aligned}
$$

Therefore, it can be simplified as follows:

$$
Tanh(x) = 2\times Sigmoid(2x) - 1
$$

Based on the derivation above, reviewing the initial Gelu formulation:

$$
Gelu(x) =0.5*x*(1 + Tanh(\sqrt {2/π}*(x + 0.044715*x^3 )))
$$

Let $Z = \sqrt{2/\pi}*(x + 0.044715 \times x^3)$. Substituting $Z$ into the expression yields the simplified form:

$$
Gelu(x) = x*Sigmoid(2Z)
$$

Expanding the formula again and applying the constant approximation $\sqrt{8/\pi} \approx 1.595769$ yields:

$$
Gelu(x) \approx x*Sigmoid (1.595769*(x + 0.044715*x^3))
$$

Expanding the $\text{Sigmoid}$ function in the expression above results in the final execution form for Gelu:

$$
\text{Gelu}(x) = \frac{x}{1 + e^{-1.595769*(x + 0.044715*x^3)}}
$$

## Example

- After obtaining the code, compile the operator executable file. For details, see [CATLASS Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Compiling a specified case
bash scripts/build.sh 27_matmul_gelu
cd output/bin
# Executable file name | Matrix M-axis | N-axis | K-axis | Device ID
# The device ID is optional. The default value is 0.
./27_matmul_gelu 256 512 1024 0
```

If the following result is displayed, precision verification is successful.

```text
Compare success.
```
