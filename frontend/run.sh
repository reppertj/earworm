#!/usr/bin/env bash

case $1 in
  start)
    # The '| cat' is to trick Node into thinking this is a non-TTY terminal
    # then react-scripts won't clear the console.
    npm start | cat
    ;;
  build)
    npm build
    ;;
  test)
    npm test $@
    ;;
  *)
    npm "$@"
    ;;
esac