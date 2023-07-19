(ns openllm.components.chat.db
  "The branch of the `app-db` that saves data related to the chat view. This
   includes the chat history, the current input value, and the layout of the
   prompt.
   The path to this branch can be expressed as:
   *root -> components -> chat*"
  (:require [cljs.spec.alpha :as s]))

(defn key-seq
  [& more-keys]
  (into [:components-db :chat-db] more-keys))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::chat-input-value string?)                                 ;; the current value of the input field
(s/def ::chat-history (s/coll-of (s/keys :req-un [::user ::text])  ;; the chat history
                                 :kind vector?))
(s/def ::layout-modal-open? boolean?)                              ;; whether the prompt layout modal is open
(s/def ::prompt-layout string?)                                    ;; the current prompt layout

(s/def ::chat-db (s/keys :req-un [::chat-input-value               ;; the spec for the chat-db
                                  ::chat-history
                                  ::layout-modal-open?
                                  ::prompt-layout]))

(defn initial-db
  []
  {:chat-input-value ""
   :chat-history []
   :prompt-layout ""
   :layout-modal-open? false})

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; check if initial-db is valid
  (s/valid? ::chat-db (initial-db)))
