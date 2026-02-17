# Agent Guidelines

## Project Structure Rules

1. **Separate Folders for Projects**: Each project will have its own dedicated folder. The user will indicate when to start a new project.

2. **Temp Folder Convention**: All temporary files for a particular project (debugging scripts, test scripts, database fetching scripts, etc.) must reside in a `temp/` folder within that project's directory.

## Planning Workflow

3. **Detailed Planning Required**: Before starting any project, we must plan it out in detail and write the plan to a `plan.md` file in the project folder.

4. **Collaborative Problem Solving**: If while following the plan:
   - The model gets stuck
   - Something is not going according to plan
   - An approach fails
   
   The model MUST stop and discuss with the user instead of continuing to find solutions independently. We will revise the plan together, and only then will the model continue execution. This loop continues until the project is complete.

## General Guidelines

- This AGENTS.md file applies to all projects in this repository
- Each project folder should have its own plan.md before work begins
- Temp files should never be committed to git
