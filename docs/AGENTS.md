# Documentation Rules

## Module Context

Project documentation including requirements, design, and implementation plans.

### Directory Structure
```
docs/
├── AGENTS.md           # This file - documentation rules
├── PRD.md              # Product Requirements Document
├── LLD.md              # Low-Level Design Document
├── PLAN.md             # Original implementation plan
└── plans/
    └── PLAN_agent_system.md  # Detailed TDD plan
```

---

## Document Standards

### Format
- All documents in Markdown format
- Use ATX-style headers (# not ===)
- Tables for structured data
- Code blocks with language specification

### Structure
Each document should have:
1. Title and metadata (date, version, status)
2. Overview/Summary section
3. Detailed content sections
4. References section

---

## Document Types

### PRD (Product Requirements Document)
- Purpose: Define WHAT to build
- Audience: All stakeholders
- Update frequency: Major feature changes
- Owner: Product/Project lead

### LLD (Low-Level Design Document)
- Purpose: Define HOW to build
- Audience: Developers
- Update frequency: Architecture changes
- Owner: Technical lead

### PLAN (Implementation Plan)
- Purpose: Define execution ORDER
- Audience: Development team
- Update frequency: Each phase completion
- Owner: Implementation lead

---

## Maintenance Rules

### When to Update
- PRD: New requirements or scope changes
- LLD: Technical design changes
- PLAN: Phase completion or blockers

### Update Process
1. Create backup of current version
2. Make changes with clear diff
3. Update metadata (date, version)
4. Commit with descriptive message

---

## Local Golden Rules

### Do's
- Date stamp all documents
- Version major changes
- Link related documents
- Keep language consistent (EN or KO)

### Don'ts
- Don't duplicate content across documents
- Don't leave TODO items in final versions
- Don't remove historical decisions
- Don't change structure without team agreement

---

## Templates Location

Plan templates are stored in:
- Feature plans: `docs/plans/PLAN_*.md`

Use existing templates as reference for new documents.
