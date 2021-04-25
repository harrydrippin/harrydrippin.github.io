FROM ruby:2.6.7
WORKDIR /srv/jekyll

COPY ./Gemfile Gemfile
RUN bundle install

EXPOSE 4000

ENTRYPOINT [ "jekyll", "serve", "--host", "0.0.0.0" ]