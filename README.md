# Seunghwan Hong's Blog

## How to build

```bash
$ docker build -t harrydrippin-blog:latest .
$ docker run --rm --name blog --volume "$PWD:/srv/jekyll" -p 4000:4000 harrydrippin-blog:latest --watch
Configuration file: /srv/jekyll/_config.yml
            Source: /srv/jekyll
       Destination: /srv/jekyll/_site
 Incremental build: disabled. Enable with --incremental
      Generating... 
                    done in 2.085 seconds.
 Auto-regeneration: enabled for '/srv/jekyll'
    Server address: http://0.0.0.0:4000/
  Server running... press ctrl-c to stop.
```
## License

The design and architecture of this blog was forked from Hux Blog.

```
Apache License 2.0.
Copyright (c) 2015-present Huxpro (huxpro.github.io)

Hux Blog is derived from [Clean Blog Jekyll Theme (MIT License)](https://github.com/BlackrockDigital/startbootstrap-clean-blog-jekyll/)
Copyright (c) 2013-2016 Blackrock Digital LLC.
```
