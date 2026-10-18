# External environment boundary

This release conveys own tool code/materials. Users provide Python, Torch, model-definition packages, native libraries and optional CUDA/driver environment. The launcher selects and checks that environment; it does not automatically create or repair it. The `llama` adapter is not llama.cpp integration.

Use exact official source/version/hash and platform/ABI compatibility when provisioning dependencies. The included CPU lock is one Linux CPython3.14 plan, not a Windows/CUDA lock. An import check does not establish driver compatibility, actual device execution or training correctness.

A future deployer should first produce an inspectable plan with destinations, download sizes/hashes, version/driver/ABI constraints, agreement requirements and rollback/removal boundaries. Nonstandard package sources or new terms require an explicit user-side step. No silent post-install networking, driver changes or acceptance flags are part of the deb. Separate installation does not itself grant third-party linking/conveying rights.
