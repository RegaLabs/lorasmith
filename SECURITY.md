# Security

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately instead:

- on GitHub, with **Report a vulnerability** under the repository's Security tab, or
- by email to contact@regalabs.dev.

Say what you found, how to reproduce it, and which version or commit you used. You should get
an answer within a week. Once a fix is released, the report is credited to you unless you
would rather it wasn't.

## What counts

lorasmith reads files that come from other people: model weights, tokenizer and config files
from Hugging Face, and the training data you point it at. A file that makes lorasmith run
code, write outside the run folder, or read files it wasn't given is a vulnerability.
Crashing or running out of memory on a damaged file is a bug: a normal issue is fine for that.

## Supported versions

Fixes go into the latest release and the `main` branch.
