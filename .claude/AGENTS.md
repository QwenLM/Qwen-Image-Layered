# Claude Agent System Rules

## Module Context

Configuration for Claude Code Skills and SubAgents.
Defines automated triggers and task delegation patterns.

### Directory Structure
```
.claude/
├── AGENTS.md           # This file - agent system rules
├── CLAUDE.md           # Project context document
├── skills/             # Auto-trigger modules
│   ├── image-decompose/
│   ├── layer-export/
│   ├── quick-edit/
│   └── background-remove/
└── agents/             # Task delegation agents
    ├── batch_processor.md
    ├── layer_editor.md
    ├── composition_engine.md
    └── quality_checker.md
```

---

## Skills vs SubAgents

### Skills (Auto-Trigger)
- Triggered automatically by keyword detection
- Handle single, well-defined tasks
- Execute immediately without delegation
- Suitable for: simple operations, quick actions

### SubAgents (Delegation)
- Invoked explicitly by main agent
- Handle complex, multi-step workflows
- Maintain state during execution
- Suitable for: batch processing, complex editing

---

## Skill Definition Rules

### File Structure
```
skills/{skill-name}/
└── SKILL.md
```

### Required Sections
1. Metadata (name, description)
2. Triggers (keywords, patterns)
3. Inputs (parameters with types)
4. Outputs (return values)
5. Execution (script or inline code)

### SKILL.md Template
```markdown
---
name: skill-name
description: One-line description
---

# Triggers
Keywords: keyword1, keyword2
Patterns:
- "pattern with {variable}"

# Inputs
- input_name (type): description [required/optional]

# Outputs
- output_name (type): description

# Execution
Script: ./script.sh
# OR
Inline: |
  python code here
```

---

## SubAgent Definition Rules

### File Structure
```
agents/{agent-name}.md
```

### Required Sections
1. Frontmatter (name, description, model, tools)
2. Role description
3. Capabilities list
4. Workflow steps
5. Error handling

### Agent Template
```markdown
---
name: AgentName
description: One-line description
model: claude-sonnet-4-20250514
tools:
  - execute
  - read
  - write
---

# Role
Detailed role description.

# Capabilities
1. Capability one
2. Capability two

# Workflow
## Step 1: Name
Description and code example.

## Step 2: Name
Description and code example.

# Error Handling
How to handle failures.
```

---

## Naming Conventions

### Skills
- Directory: lowercase with hyphens (`image-decompose`)
- File: `SKILL.md` (uppercase)

### SubAgents
- File: lowercase with underscores (`batch_processor.md`)

### Triggers
- Use natural language keywords
- Include both English and Korean if applicable
- Avoid overlapping triggers between skills

---

## Local Golden Rules

### Do's
- Keep SKILL.md under 100 lines
- Keep agent .md under 200 lines
- Test triggers before deployment
- Document all input parameters

### Don'ts
- Don't create overlapping triggers
- Don't use emojis in definitions
- Don't hardcode file paths
- Don't skip error handling sections

---

## Trigger Priority

When multiple skills match, priority order:
1. Exact keyword match
2. Pattern match with more specificity
3. First defined skill

### Conflict Resolution
If triggers conflict, modify keywords to be more specific:
- Bad: "export" (too generic)
- Good: "export to PPTX", "export layers"

---

## Testing Skills and Agents

### Manual Testing
1. Start new Claude Code session
2. Input trigger phrase
3. Verify correct skill/agent activates
4. Check output matches expected

### Checklist
- [ ] Trigger activates correctly
- [ ] Inputs are parsed properly
- [ ] Execution completes without error
- [ ] Output format is correct
