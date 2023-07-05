(ns openllm.components.apis.events
  (:require [openllm.components.apis.data :as data]
            [openllm.events :refer [check-spec-interceptor]]
            [re-frame.core :refer [reg-event-db reg-event-fx]]))

(defn api-id->http-event
  "Returns the http event for the given `api-id`."
  [api-id]
  (->> data/endpoints-data
       (filter #(= (:id %) api-id))
       first
       :event))

(reg-event-db
 ::set-selected-api
 [check-spec-interceptor]
 (fn [db [_ selected-api]]
   (assoc db :selected-api selected-api)))

(reg-event-db
 ::set-input-value
 [check-spec-interceptor]
 (fn [db [_ id value]]
   (assoc-in db [:apis-data id :input-value] value)))

(reg-event-db
 ::set-last-response
 [check-spec-interceptor]
 (fn [db [_ id value]]
   (assoc-in db [:apis-data id :last-response] value)))

(reg-event-fx
 ::on-send-button-click
 [check-spec-interceptor]
 (fn [_ [_ id value]]
   {:dispatch (into (api-id->http-event id) [value {:on-success [::set-last-response id]
                                                    :on-failure [::set-last-response id]}])}))
