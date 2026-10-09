# Contributing to AlpaJudge

We welcome external contributions to the AlpaJudge inference package. Contributions are
licensed under the [Apache License 2.0](LICENSE).

## Contribution guidelines

- Keep each pull request focused on one change and follow the existing code style.
- Include tests and documentation updates when behavior changes.
- Follow the setup instructions in [README.md](README.md#setup) and run `uv run pytest`
  from the repository root before submitting a pull request.
- Describe the problem, the change, and the validation performed in your pull request.
- Do not commit model weights, datasets, generated outputs, caches, or credentials.

## Developer Certificate of Origin

Every contributor must sign off on every commit to certify the
[Developer Certificate of Origin (DCO), version 1.1](https://developercertificate.org/).
By signing off, you certify that you have the right to submit the contribution under the
project's license. Contributions containing commits without a sign-off will not be accepted.

Use your name and email address when signing off:

```bash
git commit --signoff -m "Describe the change"
```

Git appends a trailer to the commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

To add a missing sign-off to your most recent commit:

```bash
git commit --amend --no-edit --signoff
```

Full text of the DCO:

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) the contribution was created in whole or in part by me and I have the right to submit it under the open
      source license indicated in the file; or

  (b) the contribution is based upon previous work that, to the best of my knowledge, is covered under an
      appropriate open source license and I have the right under that license to submit that work with
      modifications, whether created in whole or in part by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated in the file; or

  (c) the contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have
      not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the
      contribution (including all personal information I submit with it, including my sign-off) is maintained
      indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
