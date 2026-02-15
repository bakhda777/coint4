---
name: gsd:help
description: Show available GSD commands and usage guide
---

<objective>
Display the complete GSD command reference.

Output ONLY the reference content below. Do NOT add:

- Project-specific analysis
- Git status or file context
- Next-step suggestions
- Any commentary beyond the reference
  </objective>

<reference>
# GSD Command Reference

**GSD** (Get Shit Done) creates hierarchical project plans optimized for solo agentic development with OpenAI Codex CLI.

## Quick Start

1. `/prompts:gsd-new-project` - Initialize project (includes research, requirements, roadmap)
2. `/prompts:gsd-plan-phase 1` - Create detailed plan for first phase
3. `/prompts:gsd-execute-phase 1` - Execute the phase

## Staying Updated

GSD evolves fast. Check for updates periodically:

```
/prompts:gsd-whats-new
```

Shows what changed since your installed version. Update with:

```bash
npx get-shit-done-cc@latest
```

## Core Workflow

```
/prompts:gsd-new-project → /prompts:gsd-plan-phase → /prompts:gsd-execute-phase → repeat
```

### Project Initialization

**`/prompts:gsd-new-project`**
Initialize new project through unified flow.

One command takes you from idea to ready-for-planning:
- Deep questioning to understand what you're building
- Optional domain research (spawns 4 parallel researcher agents)
- Requirements definition with v1/v2/out-of-scope scoping
- Roadmap creation with phase breakdown and success criteria

Creates all `.planning/` artifacts:
- `PROJECT.md` — vision and requirements
- `config.json` — workflow mode (interactive/yolo)
- `research/` — domain research (if selected)
- `REQUIREMENTS.md` — scoped requirements with REQ-IDs
- `ROADMAP.md` — phases mapped to requirements
- `STATE.md` — project memory

Usage: `/prompts:gsd-new-project`

**`/prompts:gsd-map-codebase`**
Map an existing codebase for brownfield projects.

- Analyzes codebase with parallel Explore agents
- Creates `.planning/codebase/` with 7 focused documents
- Covers stack, architecture, structure, conventions, testing, integrations, concerns
- Use before `/prompts:gsd-new-project` on existing codebases

Usage: `/prompts:gsd-map-codebase`

### Phase Planning

**`/prompts:gsd-discuss-phase <number>`**
Help articulate your vision for a phase before planning.

- Captures how you imagine this phase working
- Creates CONTEXT.md with your vision, essentials, and boundaries
- Use when you have ideas about how something should look/feel

Usage: `/prompts:gsd-discuss-phase 2`

**`/prompts:gsd-research-phase <number>`**
Comprehensive ecosystem research for niche/complex domains.

- Discovers standard stack, architecture patterns, pitfalls
- Creates RESEARCH.md with "how experts build this" knowledge
- Use for 3D, games, audio, shaders, ML, and other specialized domains
- Goes beyond "which library" to ecosystem knowledge

Usage: `/prompts:gsd-research-phase 3`

**`/prompts:gsd-list-phase-assumptions <number>`**
See what Codex is planning to do before it starts.

- Shows Codex's intended approach for a phase
- Lets you course-correct if Codex misunderstood your vision
- No files created - conversational output only

Usage: `/prompts:gsd-list-phase-assumptions 3`

**`/prompts:gsd-plan-phase <number>`**
Create detailed execution plan for a specific phase.

- Generates `.planning/phases/XX-phase-name/XX-YY-PLAN.md`
- Breaks phase into concrete, actionable tasks
- Includes verification criteria and success measures
- Multiple plans per phase supported (XX-01, XX-02, etc.)

Usage: `/prompts:gsd-plan-phase 1`
Result: Creates `.planning/phases/01-foundation/01-01-PLAN.md`

### Execution

**`/prompts:gsd-execute-phase <phase-number>`**
Execute all plans in a phase.

- Groups plans by wave (from frontmatter), executes waves sequentially
- Plans within each wave run in parallel via Task tool
- Verifies phase goal after all plans complete
- Updates REQUIREMENTS.md, ROADMAP.md, STATE.md

Usage: `/prompts:gsd-execute-phase 5`

### Roadmap Management

**`/prompts:gsd-add-phase <description>`**
Add new phase to end of current milestone.

- Appends to ROADMAP.md
- Uses next sequential number
- Updates phase directory structure

Usage: `/prompts:gsd-add-phase "Add admin dashboard"`

**`/prompts:gsd-insert-phase <after> <description>`**
Insert urgent work as decimal phase between existing phases.

- Creates intermediate phase (e.g., 7.1 between 7 and 8)
- Useful for discovered work that must happen mid-milestone
- Maintains phase ordering

Usage: `/prompts:gsd-insert-phase 7 "Fix critical auth bug"`
Result: Creates Phase 7.1

**`/prompts:gsd-remove-phase <number>`**
Remove a future phase and renumber subsequent phases.

- Deletes phase directory and all references
- Renumbers all subsequent phases to close the gap
- Only works on future (unstarted) phases
- Git commit preserves historical record

Usage: `/prompts:gsd-remove-phase 17`
Result: Phase 17 deleted, phases 18-20 become 17-19

### Milestone Management

**`/prompts:gsd-new-milestone <name>`**
Start a new milestone through unified flow.

- Deep questioning to understand what you're building next
- Optional domain research (spawns 4 parallel researcher agents)
- Requirements definition with scoping
- Roadmap creation with phase breakdown

Mirrors `/prompts:gsd-new-project` flow for brownfield projects (existing PROJECT.md).

Usage: `/prompts:gsd-new-milestone "v2.0 Features"`

**`/prompts:gsd-complete-milestone <version>`**
Archive completed milestone and prepare for next version.

- Creates MILESTONES.md entry with stats
- Archives full details to milestones/ directory
- Creates git tag for the release
- Prepares workspace for next version

Usage: `/prompts:gsd-complete-milestone 1.0.0`

### Progress Tracking

**`/prompts:gsd-progress`**
Check project status and intelligently route to next action.

- Shows visual progress bar and completion percentage
- Summarizes recent work from SUMMARY files
- Displays current position and what's next
- Lists key decisions and open issues
- Offers to execute next plan or create it if missing
- Detects 100% milestone completion

Usage: `/prompts:gsd-progress`

### Session Management

**`/prompts:gsd-resume-work`**
Resume work from previous session with full context restoration.

- Reads STATE.md for project context
- Shows current position and recent progress
- Offers next actions based on project state

Usage: `/prompts:gsd-resume-work`

**`/prompts:gsd-pause-work`**
Create context handoff when pausing work mid-phase.

- Creates .continue-here file with current state
- Updates STATE.md session continuity section
- Captures in-progress work context

Usage: `/prompts:gsd-pause-work`

### Debugging

**`/prompts:gsd-debug [issue description]`**
Systematic debugging with persistent state across context resets.

- Gathers symptoms through adaptive questioning
- Creates `.planning/debug/[slug].md` to track investigation
- Investigates using scientific method (evidence → hypothesis → test)
- Survives `/clear` — run `/prompts:gsd-debug` with no args to resume
- Archives resolved issues to `.planning/debug/resolved/`

Usage: `/prompts:gsd-debug "login button doesn't work"`
Usage: `/prompts:gsd-debug` (resume active session)

### Todo Management

**`/prompts:gsd-add-todo [description]`**
Capture idea or task as todo from current conversation.

- Extracts context from conversation (or uses provided description)
- Creates structured todo file in `.planning/todos/pending/`
- Infers area from file paths for grouping
- Checks for duplicates before creating
- Updates STATE.md todo count

Usage: `/prompts:gsd-add-todo` (infers from conversation)
Usage: `/prompts:gsd-add-todo Add auth token refresh`

**`/prompts:gsd-check-todos [area]`**
List pending todos and select one to work on.

- Lists all pending todos with title, area, age
- Optional area filter (e.g., `/prompts:gsd-check-todos api`)
- Loads full context for selected todo
- Routes to appropriate action (work now, add to phase, brainstorm)
- Moves todo to done/ when work begins

Usage: `/prompts:gsd-check-todos`
Usage: `/prompts:gsd-check-todos api`

### Utility Commands

**`/prompts:gsd-help`**
Show this command reference.

**`/prompts:gsd-whats-new`**
See what's changed since your installed version.

- Shows installed vs latest version comparison
- Displays changelog entries for versions you've missed
- Highlights breaking changes
- Provides update instructions when behind

Usage: `/prompts:gsd-whats-new`

## Files & Structure

```
.planning/
├── PROJECT.md            # Project vision
├── ROADMAP.md            # Current phase breakdown
├── STATE.md              # Project memory & context
├── config.json           # Workflow mode & gates
├── todos/                # Captured ideas and tasks
│   ├── pending/          # Todos waiting to be worked on
│   └── done/             # Completed todos
├── debug/                # Active debug sessions
│   └── resolved/         # Archived resolved issues
├── codebase/             # Codebase map (brownfield projects)
│   ├── STACK.md          # Languages, frameworks, dependencies
│   ├── ARCHITECTURE.md   # Patterns, layers, data flow
│   ├── STRUCTURE.md      # Directory layout, key files
│   ├── CONVENTIONS.md    # Coding standards, naming
│   ├── TESTING.md        # Test setup, patterns
│   ├── INTEGRATIONS.md   # External services, APIs
│   └── CONCERNS.md       # Tech debt, known issues
└── phases/
    ├── 01-foundation/
    │   ├── 01-01-PLAN.md
    │   └── 01-01-SUMMARY.md
    └── 02-core-features/
        ├── 02-01-PLAN.md
        └── 02-01-SUMMARY.md
```

## Workflow Modes

Set during `/prompts:gsd-new-project`:

**Interactive Mode**

- Confirms each major decision
- Pauses at checkpoints for approval
- More guidance throughout

**YOLO Mode**

- Auto-approves most decisions
- Executes plans without confirmation
- Only stops for critical checkpoints

Change anytime by editing `.planning/config.json`

## Common Workflows

**Starting a new project:**

```
/prompts:gsd-new-project        # Unified flow: questioning → research → requirements → roadmap
/clear
/prompts:gsd-plan-phase 1       # Create plans for first phase
/clear
/prompts:gsd-execute-phase 1    # Execute all plans in phase
```

**Resuming work after a break:**

```
/prompts:gsd-progress  # See where you left off and continue
```

**Adding urgent mid-milestone work:**

```
/prompts:gsd-insert-phase 5 "Critical security fix"
/prompts:gsd-plan-phase 5.1
/prompts:gsd-execute-phase 5.1
```

**Completing a milestone:**

```
/prompts:gsd-complete-milestone 1.0.0
/clear
/prompts:gsd-new-milestone  # Start next milestone (questioning → research → requirements → roadmap)
```

**Capturing ideas during work:**

```
/prompts:gsd-add-todo                    # Capture from conversation context
/prompts:gsd-add-todo Fix modal z-index  # Capture with explicit description
/prompts:gsd-check-todos                 # Review and work on todos
/prompts:gsd-check-todos api             # Filter by area
```

**Debugging an issue:**

```
/prompts:gsd-debug "form submission fails silently"  # Start debug session
# ... investigation happens, context fills up ...
/clear
/prompts:gsd-debug                                    # Resume from where you left off
```

## Getting Help

- Read `.planning/PROJECT.md` for project vision
- Read `.planning/STATE.md` for current context
- Check `.planning/ROADMAP.md` for phase status
- Run `/prompts:gsd-progress` to check where you're up to
  </reference>
