# Taiga.jl

Tensor-product applications in isogeometric analysis.

## Running default job locally using Docker image

The default CI job runs tests and checks coverage. The custom `Julia` image
[m1ka05/julia-feather](https://hub.docker.com/repository/docker/m1ka05/julia-feather/general)
is equipped with the Feather registry and packages need for coverage processing. Using the
custom image decreases the duration time of the job.

Before pushing changes to the workflow, we can test the default job locally:

```gitlab-runner exec docker --docker-pull-policy="if-not-present" default```


## Semantic versioning, changelog, bumps and release


`CHANGELOG.md` is quasi-automatically generated if your commit messages look something like
```
added missing test for feature XYZ

Changelog: test
```
You can generate changelog sections for all pushed commits after some tag `TAG` by running
```
glab changelog generate --from TAG
```
For now, you have to manually paste the generated changelog sections to `CHANGELOG.md` and then
bump the version in `Project.toml`.

Follow semantic versioning rules as described here: https://semver.org/.

The generation of `CHANGELOG.md` is configured in `.gitlab/changelog_config.yml`.

Commit changes to `Project.toml` and  `CHANGELOG.md` with a message like
```
bumped version to v0.3.5
```
and tag this commit with an annotated tag
```
git tag -a v0.3.5 -m "tag version v0.3.5"
```
Finally, push everything to the repository.


### Conventional commit types

| Commit Type | Title                    | Description                                                                                                 | Emoji  |
| ----------- | ------------------------ | ----------------------------------------------------------------------------------------------------------- |:------:|
| `feat`      | Features                 | A new feature                                                                                               | ✨     |
| `fix`       | Bug Fixes                | A bug Fix                                                                                                   | 🐛     |
| `docs`      | Documentation            | Documentation only changes                                                                                  | 📚     |
| `style`     | Styles                   | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)      | 💎     |
| `refactor`  | Code Refactoring         | A code change that neither fixes a bug nor adds a feature                                                   | 📦     |
| `perf`      | Performance Improvements | A code change that improves performance                                                                     | 🚀     |
| `test`      | Tests                    | Adding missing tests or correcting existing tests                                                           | 🚨     |
| `build`     | Builds                   | Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)         | 🛠     |
| `ci`        | Continuous Integrations  | Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs) | ⚙️     |
| `chore`     | Chores                   | Other changes that don't modify src or test files                                                           | ♻️     |
| `revert`    | Reverts                  | Reverts a previous commit                                                                                   | 🗑     |

[source](https://github.com/pvdlg/conventional-commit-types)