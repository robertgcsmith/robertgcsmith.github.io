# Robert G. C. Smith — research website

A Jekyll website based on [Matthias Humt's template](https://github.com/hummat/hummat.github.io).

## Preview and publish

The development site is [robertgcsmith.github.io](https://robertgcsmith.github.io/). GitHub Pages builds and deploys the root of `main` automatically. Check the repository's **Actions → pages build and deployment** run before reviewing the published site. Pull requests run a separate Jekyll build check.

To preview a branch locally with Ruby and Bundler installed:

```sh
bundle install
bundle exec jekyll serve --drafts --livereload
```

Open `http://localhost:4000/`. Restart Jekyll after changing `_config.yml`.

Posts use the inherited front matter and Markdown format. Add `mathjax: true` for mathematics, `plotly: true` for Plotly, or `jquery: true` for the existing HTML figure loader. Writing, Topics and `feed.xml` update automatically when posts are published.

The repository has no custom-domain `CNAME` file. Keep `rgcsmith.com`, its DNS and GitHub Pages custom-domain settings unchanged until a separate domain migration is agreed.

## Credits

Thanks to [**Barry Clark**](https://www.barryclark.com) and his wonderful [**Jekyll Now**](https://github.com/barryclark/jekyll-now) blog template!
