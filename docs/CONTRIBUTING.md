# Contributing Guide

This guide will walk you through the step-by-step how to:

1. Request access as a contributor to the project
2. Download the project to your local machine
3. Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/)
4. Install the [pre-commit hook](https://pre-commit.com/) for this repo, which will automatically enforce standard code style
5. Get familiar with making changes in Git and on GitHub
6. Familiarize you with how to run the test suite

## Request access to contribute to the project

This should be a fairly simple email request to Sam (sjcdelap@gmail.com, Core Maintainer)
and Nicolai (nkonow@gmail.com, Lab PI) for collaboration. In this email, briefly state:

1. The program you work with
2. The reason you'd like to contribute to the project
3. Any plans/ideas for the duration of your contributions.

We are always on the lookout for additional maintainers, so please feel free to drop
us a line!

We will respond to you as soon as we can with next steps.

Once you have confirmed that you have access to the repository, you can now download it onto your local machine.


## Download the project to your local machine

This project, as most other collaborative code projects do, uses [Git](https://git-scm.com/)
and remote code storage (via GitHub) to allow for open sharing and collaboration between
groups.

### Install Git

If you haven't already, follow this [guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
to install git on your computer. This will let you access the code and modify it freely in a version-controllled
environment

### Create a personal access token

1. Follow GitHub's [guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token) for creating your first personal access token.
This will allow you to authenticate with GitHub whenever you download the repository.

### Download (clone) the repository

1. Navigate to the [repository](https://github.com/sam-delap/deepxromm).
2. Click the green "code" button above the file layout in the repostiory, then click on HTTPS
3. Copy and run the command in your terminal

## Install dependencies using uv

[uv](https://docs.astral.sh/uv/getting-started/installation/) is a modern, fast Python package and project manager.
We are going to use it to install dependencies for this project.

1. Navigate to where you had the code stored in your terminal
2. Run
    ```bash
    uv sync --dev --all-extras --locked
    ```

Congrats! You've now got all of the dependencies required to run deepxromm locally installed

## Install the pre-commit hook

This step will ensure that your code meets the style guidelines for the repository.

To install, simply run:
```bash
uv run pre-commit install
```

and the styling script will install itself on your machine. You will now be alerted
whenever your code does not match the style enforced by this repository.

## Get familiar with making changes in Git and on GitHub

Git runs using "branches", which can be thought of as versions of a codebase.
The public copy is called the "main" branch, and should only be added to after review.
This guide will teach you how to make your own branch, make changes, save them in git,
and push them to GitHub so that others can interact with them.

### Make a new branch

To make a new branch (version to make a set of changes on), run:
```bash
git checkout -b my-branch-name
```

This will create a new branch named `my-branch-name`. Now, make your changes

### Commit your changes to a branch

When you're ready to save your changes in git, run the following script from wherever you downloaded deepxromm to:
```bash
git add .
git commit -m "My-commit-message"
```

This will add all of the files you changed to the branch, and then add your changes
as a new bundle with a (hopefully) descriptive message about what you changed.  
("My-commit-message" isn't exactly all that descriptive, but you get the idea)

### Push your changes to GitHub

When you're ready to push to GitHub so that others can see your code, run:
```bash
git push
```

This may not work the first time that you run the command. Follow the prompts from the tool until you
are prompted to enter credentials. Then enter:
- Username: Your GitHub username
- Password: The GitHub personal access token you made before

### Open a pull request in GitHub

Follow this [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to create a pull request

### Pull in others' changes from GitHub

Others who are working on this may add more code to the main branch, or create new branches that you
want to look at locally in an editor. To get their code from GitHub, simply run:
```bash
git pull
```

from wherever you downloaded the project to.

## Familiarize yourself with how to run the test suite

Deepxromm provides a proactive test suite, in keeping with [BDD](https://en.wikipedia.org/wiki/Behavior-driven_development) methodology.
This is to ensure that both new contributors and seasoned maintainers alike have an easy way to know if their changes have broken
anything in the repository.

The tests will run automatically whenever you open a pull request, but you can also
run them manually for local testing by running:
```bash
uv run tests/test_deepxromm.py
```
