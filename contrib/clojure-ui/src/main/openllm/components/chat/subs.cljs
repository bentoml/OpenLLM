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

(reg-sub
 ::prompt-layout
 :<- [::components-subs/chat-db]
 (fn [chat-db _]
   (:prompt-layout chat-db)))

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
