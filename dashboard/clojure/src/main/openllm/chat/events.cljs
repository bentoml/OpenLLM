(ns openllm.chat.events
    (:require [ajax.core :as ajax]
              [openllm.events :refer [api-base-url check-spec-interceptor]]
              [re-frame.core :refer [reg-event-db reg-event-fx]]
              [clojure.string :as str]))

(reg-event-db
 ::set-chat-input-value
 [check-spec-interceptor]
 (fn [db [_ new-value]]
   (assoc db :chat-input-value new-value)))

(reg-event-fx
 ::send-prompt
 []
 (fn [_ [_ prompt llm-config]]
   {:http-xhrio {:method :post
                 :uri (str api-base-url "/v1/generate")
                 :params {:prompt prompt
                          :llm_config llm-config}
                 :format          (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success      [::send-prompt-success]
                 :on-failure      [::send-prompt-failure]}})) ;; TODO: register handler

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
     {:dispatch-n [[::send-prompt prompt llm-config]
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
