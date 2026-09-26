# Security Policy

## Supported Versions

| Version | Supported |
|:-------:|:---------:|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

The Aglibol Agent team takes security vulnerabilities seriously. Because Aglibol Agent executes local tools and manages shell commands, we maintain strict security boundaries including:
- Human-in-the-Loop (HITL) approval gates
- Workspace filesystem sandboxing
- Command execution pattern blocking and process tree isolation
- Windows case-insensitive environment token scrubbing

If you discover a security vulnerability in Aglibol Agent, please **do not** open a public issue.

### Preferred Method
Please submit your vulnerability report privately via GitHub's [Private Vulnerability Reporting](https://github.com/xushiexpresso15/AglibolAgent/security/advisories/new).

### Alternative Method
If you are unable to use GitHub Security Advisories, email your findings to:
`security@aglibol.ai`

### What to Include in Your Report
- A description of the vulnerability and its potential impact.
- Step-by-step instructions or proof-of-concept (PoC) code to reproduce the issue.
- Your assessment of the severity and affected versions.

### Our Commitment
- We will acknowledge receipt of your report within 48 hours.
- We will provide regular updates regarding our progress toward a fix.
- Once a fix is verified, we will issue a patched release and credit the reporter (unless you request anonymity).
