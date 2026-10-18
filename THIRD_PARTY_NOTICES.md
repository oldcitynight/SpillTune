# External references and runtime boundary

The project references Transformers5.13.0 architecture/configuration implementations. Exact immutable source URLs, hashes and observed header statements are retained in [legal/upstream-references.json](legal/upstream-references.json), [NOTICE](NOTICE), and the unchanged [Apache-2.0 text](LICENSES/transformers-APACHE-2.0.txt).

PyTorch, Transformers, Safetensors, Tokenizers, Python and any CUDA components are external dependencies, not vendored distributions in the own wheel/source/deb. The optional Linux/CPython3.14 CPU lock preserves exact dependency URLs and hashes for user provisioning; it does not distribute those wheels or clear every possible runtime combination. No llama.cpp integration or automatic environment deployer is implemented.

Retain and review the actual component notices/terms if dependency wheels, container images, models, tokenizers or training assets are conveyed in a future distribution. See [LICENSE.md](LICENSE.md) for the unchanged project selection and scope.
