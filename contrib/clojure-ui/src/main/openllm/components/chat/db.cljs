(ns openllm.components.chat.db
  "The branch of the app-db that saves data related to the chat view. This
   includes the chat history, the current input value, and the layout of the
   prompt.
   The path to this branch can be expressed as:
   *root -> components -> chat*"
  (:require [cljs.spec.alpha :as s]))

(defn key-seq
  [last-key]
  [:components-db :chat-db last-key])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::chat-input-value string?)
(s/def ::chat-history (s/coll-of (s/keys :req-un [::user ::text]) :kind vector?))
(s/def ::layout-modal-open? boolean?)
(s/def ::prompt-layout string?)

(s/def ::chat-db (s/keys :req-un [::chat-input-value
                                  ::chat-history
                                  ::layout-modal-open?
                                  ::prompt-layout]))

(def initial-db
  {:chat-input-value ""
   :chat-history []
   :prompt-layout ""
   :layout-modal-open? false})

(comment
  ;; check if initial-db is valid
  (s/valid? ::chat-db initial-db))
