#!/usr/bin/env bash
set -Eeuo pipefail

# check to see if this file is being run or sourced from another script
_is_sourced() {
	# https://unix.stackexchange.com/a/215279
	[ "${#FUNCNAME[@]}" -ge 2 ] &&
		[ "${FUNCNAME[0]}" = '_is_sourced' ] &&
		[ "${FUNCNAME[1]}" = 'source' ]
}

_main() {
	# For backwards compatibility with the yatai<1.0.0, adapting the old "yatai" command to the new "start" command.
	if [ "${#}" -gt 0 ] && [ "${1}" = 'python' ] && [ "${2}" = '-m' ] && { [ "${3}" = 'bentoml._internal.server.cli.runner' ] || [ "${3}" = "bentoml._internal.server.cli.api_server" ]; }; then # SC2235, use { } to avoid subshell overhead
		if [ "${3}" = 'bentoml._internal.server.cli.runner' ]; then
			set -- bentoml start-runner-server "${@:4}"
		elif [ "${3}" = 'bentoml._internal.server.cli.api_server' ]; then
			set -- bentoml start-http-server "${@:4}"
		fi
	# If no arg or first arg looks like a flag.
	elif [[ "$#" -eq 0 ]] || [[ "${1:0:1}" =~ '-' ]]; then
		# This is provided for backwards compatibility with places where user may have
		# discover this easter egg and use it in their scripts to run the container.
		if [[ -v BENTOML_SERVE_COMPONENT ]]; then
			echo "\$BENTOML_SERVE_COMPONENT is set! Calling 'bentoml start-*' instead"
			if [ "${BENTOML_SERVE_COMPONENT}" = 'http_server' ]; then
				set -- bentoml start-http-server "$@" "$BENTO_PATH"
			elif [ "${BENTOML_SERVE_COMPONENT}" = 'grpc_server' ]; then
				set -- bentoml start-grpc-server "$@" "$BENTO_PATH"
			elif [ "${BENTOML_SERVE_COMPONENT}" = 'runner' ]; then
				set -- bentoml start-runner-server "$@" "$BENTO_PATH"
			fi
		else
			set -- bentoml serve "$@" "$BENTO_PATH"
		fi
	fi
	# Overide the BENTOML_PORT if PORT env var is present. Used for Heroku and Yatai.
	if [[ -v PORT ]]; then
		echo "\$PORT is set! Overiding \$BENTOML_PORT with \$PORT ($PORT)"
		export BENTOML_PORT=$PORT
	fi
	# Handle serve and start commands that is passed to the container.
	# Assuming that serve and start commands are the first arguments
	# Note that this is the recommended way going forward to run all bentoml containers.
	if [ "${#}" -gt 0 ] && { [ "${1}" = 'serve' ] || [ "${1}" = 'serve-http' ] || [ "${1}" = 'serve-grpc' ] || [ "${1}" = 'start-http-server' ] || [ "${1}" = 'start-grpc-server' ] || [ "${1}" = 'start-runner-server' ]; }; then
		exec bentoml "$@" "$BENTO_PATH"
	else
		# otherwise default to run whatever the command is
		# This should allow running bash, sh, python, etc
		exec "$@"
	fi
}

if ! _is_sourced; then
	_main "$@"
fi
