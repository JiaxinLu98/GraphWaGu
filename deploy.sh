#!/usr/bin/env sh

set -e

# pnpm build

cd dist

git init
if ! git remote | grep -q 'origin'; then
    git remote add origin git@github.com:harp-lab/GraphWaGu.git
fi
git pull
git checkout gh-pages
git add -A
git commit -m 'deploy'

git push -f

cd -