# Copilot Coding Agent: Custom Instructions for CUDA/GPU Compilation Errors

Purpose
- When asked to “fix a CUDA/GPU compilation error,” the goal is to produce a minimal, targeted code change without compiling or running the project.
- Operate strictly via static analysis, using the error text, file paths, and line numbers provided by the user or present in the repository (e.g., logs in docs/tests).

Hard Rules (Non-Negotiable)
1. Do NOT compile or run the code.
   - Do not invoke nvcc, clang, gcc, cmake, make, ninja, bazel, python setup.py, or any build/test command.
   - Do not assume access to GPUs, drivers, or CUDA toolkits in CI.
2. Keep changes minimal and localized.
   - Only modify the files/lines implicated by the provided compiler diagnostics or directly related call sites and declarations.
   - Do not refactor unrelated code or alter public APIs unless the error is directly about them.
3. Avoid build-system edits unless explicitly required by the error.
   - Do not change CMakeLists.txt, Makefiles, Bazel files, setup.py, or CI configs unless the compilation error clearly indicates a missing include path/definition that cannot be resolved within code.
   - Do not bump CUDA versions, architectures, or flags.
4. No new dependencies.
   - Do not introduce new external libraries or submodules.
5. Preserve semantics.
   - Fix the compile error without altering the intended runtime behavior. When uncertain, choose the safest minimal change and explain the trade-off in the PR body.

Workflow the Agent Must Follow
1. Read the error carefully.
   - Extract file path(s), line number(s), and the exact error text.
   - Classify the error (annotation mismatch, kernel launch syntax, device-incompatible construct, missing include, name lookup, template instantiation, etc.).
2. Locate the code region(s).
   - Open the implicated file(s) and navigate to the exact lines referenced by the error.
   - If the error involves a declaration/use mismatch, find the corresponding declaration/definition.
3. Propose the smallest viable fix.
   - Prefer a one- or few-line change over broader refactoring.
   - When multiple solutions exist, choose the one requiring the fewest edits and least risk.
4. Validate by static reasoning only.
   - Check for consistent qualifiers, signatures, includes, and types.
   - Ensure any introduced macros or guards resolve the error on likely CUDA versions without breaking host code.
5. Document clearly in the PR.
   - Include an “Error excerpt,” “Root cause,” “Targeted fix,” and “Risk/Notes.”
   - Add before/after snippets when helpful.

Allowed Fix Patterns (CUDA/GPU Specific)
- Add or correct CUDA qualifiers:
  - Add missing __host__/__device__/__global__ annotations where the error indicates host/device usage mismatch.
  - Ensure __global__ functions return void and have correct parameter types.
- Replace non-device-compatible constructs in device code:
  - Replace std::… calls with device-friendly equivalents (e.g., sinf/cosf from <math_functions.h> or ::sinf/::cosf) when errors indicate unavailable functions in device code.
  - Remove or guard exceptions, RTTI, iostreams, and non-device lambdas from device paths.
- Include fixes:
  - Add missing includes such as <cuda_runtime.h>, <device_functions.h>, <cuda_fp16.h>, <math_functions.h>, <thrust/...> when the error shows unresolved identifiers/types.
- Name lookup and linkage issues:
  - Harmonize declarations/definitions (e.g., inline/constexpr/visibility), adjust templates to match instantiations implicated by the error.
  - Provide explicit template instantiations when the error is about undefined references to templated kernels/functions used in translation units.
- Kernel launch and grid/block issues:
  - Correct <<<grid, block, shared, stream>>> syntax errors found in diagnostics.
  - Avoid changing launch parameters unless the error is purely syntactic or type-related.
- Shared memory and address space:
  - Fix extern __shared__ declarations when the error indicates mis-declaration or scope issues.
- Atomics and intrinsics:
  - Include correct headers and use the correct overloads or intrinsics for the types indicated in the error.

Conditionals and Guards
- Prefer preprocessor guards over build flag edits when dealing with CUDA-version or architecture differences.
  - Example patterns:
    - #ifdef __CUDA_ARCH__ for device-only code paths.
    - #if defined(__CUDACC__) to gate CUDA-specific includes/qualifiers.
    - Version checks if error text indicates deprecated APIs across versions (guard with comments referencing the error).

Forbidden Actions
- No build commands of any kind.
- No modifying or adding CI jobs to “try” a build.
- No CUDA version/arch flag changes (e.g., -arch, --generate-code, -std, -Xcompiler) unless the error explicitly demands a language mode change that cannot be fixed in code, and even then prefer code-local fixes.
- No large-scale refactors or stylistic rewrites.
- No performance “tuning” changes unrelated to the compilation error.

Pull Request Requirements
- Title format:
  - Fix CUDA compile error: <short error summary> in <file:line>
- Body sections:
  1. Summary
     - One or two sentences describing the issue and the minimal fix.
  2. Error excerpt
     - Quote the exact compiler message with file and line numbers.
  3. Root cause
     - Brief explanation of why the compiler complained.
  4. Targeted fix
     - What changed, where, and why this is the smallest safe change.
  5. Alternatives considered
     - Mention at most 1–2 other options and why they were not chosen.
  6. Risk/Notes
     - Any impact on host/device behavior; version/arch guards if used.
  7. Validation (static only)
     - Describe how the change resolves the diagnostic by inspection.
- Diff hygiene:
  - Only include files directly implicated by the error.
  - Avoid formatting-only noise.

Common CUDA Error Patterns and Preferred Minimal Fixes
- “calling a __host__ function from a __device__/__global__ function” or similar:
  - Add __host__ __device__ to the callee if safe and header-local; otherwise, provide a device-compatible alternative under #ifdef __CUDA_ARCH__.
- “identifier is undefined in device code”:
  - Add the correct header or use device-available overloads (math intrinsics); avoid introducing new build flags.
- “no kernel image is available for execution on the device” (compile-time form):
  - If this appears as a compile-time diagnostic tied to code, prefer guards or overload adjustments; do not change architectures/flags here.
- Kernel launch syntax errors:
  - Correct angle bracket launch syntax or parameter types; avoid changing launch configs unless the error is purely type/syntax related.
- “unresolved extern” for templated kernels:
  - Ensure definitions are visible (header-inline) or add explicit instantiations for the used template parameters.

When Build-System Edits Are Allowed
- Only if the error explicitly references missing include directories or macro definitions that cannot be added locally in code without breaking separation of concerns.
- Even then, prefer adding an include or forward declaration in code if sufficient.
- If a build edit is unavoidable, make the smallest possible change and explain why a code-local fix is infeasible.

Checklist Before Submitting PR
- [ ] I did not compile or run the code.
- [ ] The change is minimal and touches only error-implicated code.
- [ ] I did not change build files or flags (unless explicitly justified by the error).
- [ ] I preserved runtime semantics and documented any caveats.
- [ ] PR body includes Error excerpt, Root cause, Targeted fix, and Validation (static).

Notes for Reviewers
- Expect strictly minimal changes that address the reported diagnostic.
- If subsequent errors emerge post-merge, they should be handled in separate, similarly minimal PRs to maintain clear causality and revertability.

End of file.
