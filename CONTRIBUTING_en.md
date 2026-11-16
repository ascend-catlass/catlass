# Contribution Guide

Developers are welcome to experience and contribute to this project. Before contributing to the community, please refer to the [cann-community](https://gitcode.com/cann/community) to understand the code of conduct, sign the [CLA](https://gitcode.com/cann/community#%E7%AD%BE%E7%BD%B2cla), and learn about the contribution process of the source code repository.

Developers need to pay special attention to the following points when preparing local code and submitting PRs:

1. When submitting a PR, fill in the business background, purpose, and solution of the PR carefully based on the PR template.
2. If your modification is not a simple bug fix, but involves adding new features, new interfaces, new configuration parameters or modifying code processes, please be sure to first discuss the solution through an [Issue](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4issue%E5%A4%84%E7%90%86issue%E4%BB%BB%E5%8A%A1) to avoid your code being rejected for merging. If you are not sure whether the modification can be classified as a simple bug fix, you can submit an issue for discussion.

Run local code checks before submitting to avoid PRs being blocked by formatting or static-analysis issues.

Install and register the git hook on first use:

```bash
pip install pre-commit
pre-commit install
```

After that, every `git commit` triggers the checks automatically. You can also run them manually:

```bash
pre-commit run --all-files   # check all files
pre-commit run               # check staged files only
```

Developer contribution scenarios include:

- Fixing Bugs

  If you find some bugs in this project and want to fix them, you are welcome to create an issue for feedback and tracking.

  You can create a `Bug-Report` issue to describe the bug according to the [Submitting an Issue/Handling an Issue Task](https://gitcode.com/cann/community#Submitting-an-Issue/Handling-an-Issue-Task) guide, and then enter "/assign" or "/assign @yourself" in the comment box to assign the Issue to yourself for handling.

- Contributing New Operator Examples

  If you have ideas for generalization enhancement/performance optimization for the reference implementation of a certain operator during the use of CATLASS, or have implemented a certain type of new operator and want to contribute it as a reference example, you are welcome to contribute example operators.

  You can create a `Requirement/Suggestion Issue` to describe the new sample operator and provide your design solution by following the instructions in [Submitting an Issue/Handling an Issue Task](https://gitcode.com/cann/community#Submitting-an-Issue/Handling-an-Issue-Task).
  Enter /assign or /assign @yourself in the comment box to assign the issue to you for processing.

  Place the new operator sample in the `${id}_${op_name}` subdirectory under the `examples/` directory. `${id}` indicates the unique ascending sequence, and `${op_name}` indicates the name of the new operator.

  ```bash
  catlass/examples # Main directory of operator samples
  ├── ${id}_${op_name} # Single-operator sample directory
  │ ├── CMakeLists.txt # CMake build file
  │ ├── README.md # Single-operator sample description document
  │ └── ${op_name}.cpp # Main file
  ...
  ```

  The `README.md` file provides a brief description of the operator, including code organization and usage examples (for details, see the [README file](https://gitcode.com/cann/catlass/blob/master/examples/00_basic_matmul/README.md) of Matmul).

- Contributing CATLASS Code

  If you have new feature implementations for a certain part of CATLASS code, you are welcome to propose new ideas and designs in an Issue.

  You can create a `Requirement|Suggestion` Issue to describe the new feature design by following the instructions in [Submitting an Issue/Handling an Issue Task](https://gitcode.com/cann/community#Submitting-an-Issue/Handling-an-Issue-Task). Project members will communicate with you to confirm the design and provide a suitable location under `include/catlass/` for you to contribute your feature code.

  In addition, you need to comment /assign or /assign @yourself in the submitted Issue to claim the Issue and complete the code upload in the future.

- Reporting Document Error

  If you find any errors or unclear descriptions in the documents of this project, you are welcome to create an Issue for feedback and correction.

  You can create a `Documentation|Document Feedback` Issue to point out the problem in the corresponding document by following the instructions in [Submitting an Issue/Handling an Issue Task](https://gitcode.com/cann/community#Submitting-an-Issue/Handling-an-Issue-Task). Then, enter /assign or /assign @yourself in the comment box to assign the Issue to yourself for correcting the document description.

- Helping Resolve Others' Issues

  If you have a suitable solution to a problem encountered by others in the community, you are welcome to comment and communicate in the Issue to help others solve problems and pain points, and jointly optimize ease of use.

  If the corresponding Issue requires code modification, you can enter /assign or /assign @yourself in the Issue comment box to assign the Issue to yourself and track it to assist in solving the problem.

## Notes

- When submitting an Issue, follow the template under [`.gitcode/ISSUE_TEMPLATE`](.gitcode/ISSUE_TEMPLATE). If an Issue does not follow a template, the Committer asks for the missing information in a comment. An Issue that still does not meet the requirements one week after submission is closed, with the applicable ground stated in the closure comment.

- The following components are handled at low priority: non-essential modifications are closed directly. If you need to contribute to these components, attend the biweekly SIG meeting for discussion first.
  - [catlass_cppgen](python/catlass_cppgen/README.md): a Python calling solution from before CATLASS Python was available. Only a small number of community projects are known to use it, its feature scope is limited, and changes may make those projects unusable.
  - [mstuner_catlass](tools/tuner/README.md): the tuning tool builds a search space, compiles a large number of Kernels and runs them in batches to collect performance data, which is equivalent to repeatedly modifying the Tiling in a sample by hand and then collecting performance with msprof to find the best Tiling. AI agents can handle such repetitive work well, and do not need to first integrate the operator into the framework, whereas using this tool requires that integration and adds extra work.

- The following cases are also closed with the applicable ground stated in the closure comment: the problem description, environment information, reproduction steps, or logs are missing and are still not provided one week after the maintainer's first request; the Issue describes the same problem as an existing Issue, in which case it is closed and linked to the original Issue; the issue is already fixed on master, in which case it is closed and linked to the corresponding commit.

- These rules also apply to pull requests. A pull request without a linked Issue and without an explanation is handled as insufficient information.

- Submissions that appear to be generated by scripts in bulk, or a large number of homogeneous Issues or pull requests submitted by the same account in a short period, are closed as a batch with the ground stated, and are reported to the community.
