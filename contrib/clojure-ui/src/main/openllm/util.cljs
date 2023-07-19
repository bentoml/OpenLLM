(ns openllm.util
  "A collection of utility functions used throughout the application.
   All functions of this namespace must be pure."
  (:require [clojure.string :as str]))

(defn chat-history->string
  "Converts a chat history to a string representation. Basically, it joins
   the user name and the text of each entry with a colon. The entries are
   separated by a newline.
   Useful for building a chat prompt or exporting the chat history to a
   file.
   Returns the chat history as a string."
  [chat-history]
  (let [entry->chat-line (fn [current-entry]
                           (str/join ": "
                                     [(name (:user current-entry))
                                      (:text current-entry)]))]
    (str/join "\n" (map entry->chat-line
                        chat-history))))
