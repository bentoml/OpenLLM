(ns openllm.components.playground.events
  (:require [openllm.events :refer [check-spec-interceptor]]
            [re-frame.core :as rf :refer [reg-event-db reg-event-fx]]
            [openllm.api.http :as api]
            [openllm.api.log4cljs.core :refer [log]]))

(reg-event-db
 ::set-prompt-input
 [check-spec-interceptor]
 (fn [db [_ value]]
   (assoc db :playground-input-value value)))

(reg-event-fx
 ::send-prompt-success
 [check-spec-interceptor]
 (fn [db [_ response]]
   {:db (assoc db :playground-last-response (first (:responses response)))
    :dispatch [::toggle-modal]}))

(reg-event-fx
 ::send-prompt-failure
 [check-spec-interceptor]
 (fn [{:keys [db]} [_ e]]
   (log :error "Failed to send prompt" e)
   {:db (assoc db :playground-last-response "Sorry, something went wrong.")
    :dispatch [::toggle-modal]}))

(reg-event-fx
 ::on-send-button-click
 []
 (fn [_ [_ prompt llm-config]]
   {:dispatch [::api/v1-generate prompt llm-config {:on-success [::send-prompt-success]
                                                    :on-failure [::send-prompt-failure]}]}))

(reg-event-db
 ::toggle-modal
 []
 (fn [db _]
   (assoc-in db [:modal-open? :playground] (not (get-in db [:modal-open? :playground])))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
 (comment
  ;; clear input field
   (rf/dispatch [::set-prompt-input ""]))
