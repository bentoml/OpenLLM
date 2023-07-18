(ns openllm.util
  "A collection of utility functions used throughout the application.
   All functions of this namespace must be pure."
  (:require [clojure.string :as str]))

(defn chat-history->string
  "Converts a chat history to a string representation."
  [chat-history]
  (let [entry->chat-line (fn [current-entry]
                           (str/join ": "
                                     [(name (:user current-entry))
                                      (:text current-entry)]))]
    (str/join "\n"
              (map entry->chat-line
                   chat-history))))
