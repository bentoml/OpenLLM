(ns openllm.components.chat.subs
  (:require [openllm.components.subs :as components-subs]
            [openllm.util :as util]
            [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-input-value
 :<- [::components-subs/chat-db]
 (fn [chat-db _]
   (:chat-input-value chat-db)))

(reg-sub
 ::chat-history
 :<- [::components-subs/chat-db]
 (fn [chat-db _]
   (:chat-history chat-db)))

(reg-sub
 ::modal-open?
 :<- [::components-subs/chat-db]
 (fn [chat-db _]
   (:layout-modal-open? chat-db)))

;; This subscription informs it's subscribers about the current prompt layout and
;; possible changes. The user can freely choose any prompt layout, right now it
;; only acts as some kind of preamble.
(reg-sub
 ::prompt-layout
 :<- [::components-subs/chat-db]
 (fn [chat-db _]
   (:prompt-layout chat-db)))

;; This subscription materializes all the data neccessary to build a prompt for a
;; chat model.
;; In essence, it is a concatenation of the prompt layout, the chat history, and
;; the current input value. Lastly we indicate to the model, that it should keep
;; generating tokens from the AI persona.
;; TODO: The names should probably be configurable by the user in the future.
(reg-sub
 ::prompt
 :<- [::prompt-layout]
 :<- [::chat-input-value]
 :<- [::chat-history]
 (fn [[prompt-layout chat-input-value chat-history] _]
   (let [conversation (util/chat-history->string chat-history)]
     (str prompt-layout "\n"
          conversation "\n"
          "user: " chat-input-value "\n"
          "model: "))))
