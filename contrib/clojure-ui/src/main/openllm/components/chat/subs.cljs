(ns openllm.components.chat.subs
    (:require [openllm.util :as util]
              [re-frame.core :refer [reg-sub]]
              [clojure.string :as str]))

(reg-sub
 ::chat-input-value
 (fn [db _]
   (:chat-input-value db)))

(reg-sub
 ::chat-history
 (fn [db _]
   (:chat-history db)))

(reg-sub
 ::modal-open?
 :<- [:modal-open?-map]
 (fn [modal-open?-map _]
   (get modal-open?-map :chat)))

(reg-sub
 ::prompt-layout
 (fn [db _]
   (:prompt-layout db)))

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
