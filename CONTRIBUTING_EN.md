## Welcome!

If you are new to pyqpanda-algorithm (referred to as "alg") and interested in open source and community contributions, we welcome you to join us anytime! There are always opportunities to improve documentation (like the one you’re reading), review code, refactor/annotate functions/classes, or supplement test cases. We will help you understand pyqpanda-algorithm and guide you to make your first contribution smoothly!

### The alg Community’s Commitment

To foster an open and inclusive environment, the community is committed to providing a harassment-free experience for everyone—regardless of age, physical condition, body size, disability, ethnicity, sex characteristics, gender identity/expression, experience level, education, socioeconomic status, nationality, personal appearance, race, religion, or sexual orientation.

#### Code of Conduct

- **Examples of positive behavior**:
  - Use welcoming and inclusive language
  - Respect diverse viewpoints and experiences
  - Gracefully accept constructive criticism
  - Focus on what is best for the community
  - Show empathy to other community members

- **Examples of unacceptable behavior**:
  - Use sexualized language/imagery or unwelcome sexual attention/advances
  - Make racial/political insinuations
  - Troll, use insulting/derogatory language, or launch personal/political attacks
  - Harass others (publicly or privately)
  - Publish others’ private information without explicit permission
  - Other conduct deemed inappropriate in a professional setting

#### Maintainer Responsibilities

Project maintainers are responsible for clarifying acceptable/unacceptable behavior standards and must take appropriate, fair corrective action in response to unacceptable behavior.  
Maintainers have the right to remove/edit/reject comments, commits, code, wiki edits, issues, and other contributions that violate this Code of Conduct. They may also temporarily/permanently ban contributors who threaten, offend, or harm the community.

#### Oversight & Reporting

Instances of abuse, harassment, or other unacceptable behavior may be reported to [qcloud@originqc.com](mailto:qcloud@originqc.com). All complaints will be reviewed and investigated, and a necessary, appropriate response will be provided. The project team is obligated to protect the confidentiality/privacy of reporters.

### Understanding the alg Community

The alg community is built on Origin Quantum’s open-source pyqpanda-algorithm repository, which includes 13 common quantum algorithms. The repository is organized by algorithm type to streamline management and workflow improvements. We provide contribution guidelines and a series of issues for community members—everyone is welcome to join and participate in discussions.

- Community members can drive the delivery of outcomes and strive to include them in community releases.
- Contributors can gain experience and enhance their influence.
- "alg" refers to a single Repository where all deliverables are stored.
- You can submit Issues, participate in discussions, fix problems, and join code reviews within alg.

### Three Steps to Engage with the Repository

- **Enable "Watching"**：elect "all activity" to receive real-time updates on the repository’s issues, PRs, and releases—never miss important messages.

- **Star the Repository**：Starring makes it easier to find the repository later and shows your support for the alg community!

- **Fork the Repository**：Create an independent copy in your personal repository to experiment with code freely (without affecting the original project).

  - Prepare the development environment

  - Download and build the package

  - Modify the build and verify locally

  - Complete multiple commits

  - Submit a PR for merging

After completing these three steps, you’ve officially supported the alg community—let’s start contributing!  

## Start Your Contribution

We welcome contributions via submitting/resolving Issues or opening Pull Requests (PRs)! Accepted contributions include (but are not limited to): code submissions, bug fixes, documentation improvements, example supplements, feature suggestions, etc.

### Submit/Resolve an Issue

#### **Find an Issue You’re Interested In**

- **Method 1**: Browse the issue list to explore all open issues and locate your area of interest.
- **Method 2**: Use fuzzy search (by issue name) in the issue list to quickly find relevant issues.

If you cannot find the issue you want, send an email to [qcloud@originqc.com](mailto:qcloud@originqc.com) with the subject "[Question about Issue Process]" and describe the issue’s characteristics—we will assist you.

####  **Create an Issue**

To report a bug, suggest a feature, or raise a question, create an Issue following these guidelines:

- **Confirm Prerequisites**: Search for similar issues in the community/latest algorithm version to avoid duplication.
- **Create the Issue**: On the project’s GitHub page, go to the `Issues` tab → click `New issue` → select the appropriate label to associate with the relevant module.
- **Fill in a Clear Description**:
  - **Title**: Summarize the problem clearly (e.g., "Update outdated configuration item in documentation").
  - **Description**:
    - **Problem/Feature Request**: Briefly describe the issue or feature.
    - **Reproduction Steps/Expected Solution**:
      - For bugs/errors: List detailed reproduction steps and explain the gap between expected and actual results.
      - For feature requests: Describe the problem to solve, solution approach, or desired feature behavior.
  - **Supplementary Materials**: Attach logs, screenshots, or error messages (if available).
  - **Associate Existing Issues**: Link related issues with `#` (e.g., `#123`).
  - **Add Labels**: Use labels like `/bug`, `/documentation`, `/help wanted`, `/New Request` to facilitate retrieval.
  - **Participate in Discussions**: Share your opinions in the issue’s comment section.

#### **Assign/Claim an Issue**

To claim an issue (e.g., `good first issue`, `enhancement issue`):

1. In the alg issue list, filter by Labels (check `good first issue`/`help wanted`/`enhancement issue`).
   - `good first issue`: Easy to solve (ideal for beginners) – covers documentation/algorithm reasoning with clear participation guidelines.
   - `help wanted`: Community-driven issues requiring more time/discussion (suitable for experienced contributors).
   - `enhancement issue`: Complex issues (e.g., algorithm cases, code optimization) – ideal for hackathons/internships.
2. Enter the issue page and comment `/assign` or `/assign @yourself` to claim it.
3. Your name will appear in the assignee list once the claim is successful.

### Submit a Pull Request (PR)

Before submitting a PR, ensure:

- Code complies with the project’s style and passes CI tests (run `tox` locally to check). 
- Relevant documentation (including docstrings) is updated.
- Additional tests are added for impactful changes.
- Release notes are added for user-facing changes (mark the PR as a changelog).

#### PR Workflow

1. Fork the alg repository ([repo page](https://github.com/OriginQ/pyqpanda-algorithm)) and clone it locally.

2. Create a new branch from `develop`:  
   `git checkout develop -b new_branch_name`

3. Commit your changes to the new branch.

4. Sync your forked `develop` with the official repository (resolve merge conflicts if needed):

   ```bash
   # Update local develop
   git fetch upstream 
   git checkout develop 
   git merge upstream/develop 
   # Merge develop into your branch
   git checkout new_branch_name 
   git merge develop
   ```

### Testing

Write unit tests for modified code and pass all existing tests before submitting a PR.

#### Testing Guidelines

1. Unit test files must be named `[feature].test.py`.
2. Use `Test([feature], [test_content])` for test case naming.
3. Follow pyqpanda3’s code standards for unit test code.

## Acknowledgements

Thanks to all contributors, testers, and community supporters. Special thanks to the Origin Quantum Research Institute for technical support in algorithm design and performance optimization.

------

## Contact Us

- **Official Email**：[qcloud@originqc.com](mailto:qcloud@originqc.com)

- **Pre-sales Consultation**：https://contact.originqc.com.cn/

- **Official WeChat**：Search for "本源量子云社区"，follow Open-Source Project Updates.
<p align="center">
  <img src="my-folder/服务号.png" alt="本源量子云社区服务号" width="50%">
</p>

- **The official assistant**：scan the QR code below to add the official assistant for quantum cloud support and event updates
<p align="center">
  <img src="my-folder/本源量子云小助手.jpg" alt="本源量子官方小助手" width="30%">
</p>

