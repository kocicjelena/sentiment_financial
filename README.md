git lfs track "*.bin*"

git lfs track "models/roberta-base"
git add .gitattributes
git lfs migrate info --everything --include="*.bin"

# perform migration
git lfs migrate import --everything --include="*.bin" --verbose
git lfs push --all origin