* Check that all unit tests are OK
* Run the demo script and listen to the sounds to empirically check the results
* Bump the version number in `torch_audiomentations/__init__.py` in accordance with the [semantic versioning specification](https://semver.org/)
* Write a summary of the changes in the changelog section in README.md
* Commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
* Add and push a git tag to the release commit
* Add a release here: https://github.com/asteroid-team/torch-audiomentations/releases/new
* Update the Zenodo badge in README.md. Commit and push.
* `python setup.py sdist bdist_wheel`
* `python -m twine upload dist/*`
