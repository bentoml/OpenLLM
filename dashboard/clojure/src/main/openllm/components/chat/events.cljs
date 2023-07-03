(ns openllm.components.chat.events
    (:require [openllm.events :refer [check-spec-interceptor]] 
              [openllm.api.http :as api]
              [re-frame.core :refer [reg-event-db reg-event-fx]]
              [clojure.string :as str]))

(reg-event-db
 ::set-chat-input-value
 [check-spec-interceptor]
 (fn [db [_ new-value]]
   (assoc db :chat-input-value new-value)))

(reg-event-db
 ::add-to-chat-history
 [check-spec-interceptor]
 (fn [db [_ user text]]
   (assoc db :chat-history (conj (:chat-history db) {:user user
                                                     :text text}))))

(reg-event-fx
 ::send-prompt-success
 []
 (fn [_ [_ response]]
   {:dispatch [::add-to-chat-history :model (first (:responses response))]}))

(reg-event-fx
 ::send-prompt-failure
 []
 (fn [_ [_ _]]
   {:dispatch-later [{:ms 10 :dispatch [::add-to-chat-history :model "Sorry, something went wrong."]}
                     {:ms 20 :dispatch [::auto-scroll]}]}))

(reg-event-fx
 ::on-send-button-click
 []
 (fn [_ [_ prompt llm-config]]
   (when (not (str/blank? prompt))
     {:dispatch-n [[::api/v1-generate prompt llm-config {:on-success [::send-prompt-success]
                                                         :on-failure [::send-prompt-failure]}]
                   [::add-to-chat-history :user prompt]
                   [::set-chat-input-value ""]]
      :dispatch-later [{:ms 20 :dispatch [::auto-scroll]}]})))

(defn auto-scroll!
  "Scrolls the chat history to the bottom. Has side effects obviously."
  []
  (let [chat-history (js/document.getElementById "chat-history-container")]
    (set! (.-scrollTop chat-history) (.-scrollHeight chat-history))))

(reg-event-fx
 ::auto-scroll
 []
 (fn [_ _]
   (auto-scroll!) {}))
