(ns openllm.components.playground.events
    (:require [openllm.events :refer [check-spec-interceptor]]
              [re-frame.core :as rf :refer [reg-event-db reg-event-fx]]
              [openllm.api.events :as api]))

(reg-event-db
 ::set-prompt-input
 [check-spec-interceptor]
 (fn [db [_ value]]
   (assoc db :playground-input-value value)))

(reg-event-db
 ::send-prompt-success
 [check-spec-interceptor]
 (fn [db [_ response]]
   (assoc db :playground-last-response (first (:responses response)))))

(reg-event-db
 ::send-prompt-failure
 [check-spec-interceptor]
 (fn [db [_ _]]
   (assoc db :playground-last-response "Sorry, something went wrong.")))

(reg-event-fx
 ::on-send-button-click
 []
 (fn [_ [_ prompt llm-config]]
   {:dispatch [::api/v1-generate prompt llm-config {:on-success [::send-prompt-success]
                                                    :on-failure [::send-prompt-failure]}]}))
