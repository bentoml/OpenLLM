(ns openllm.components.apis.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::selected-api
 (fn [db _]
   (:selected-api db)))

(reg-sub
 ::input-value
 (fn [db _]
   (get-in db [:apis-data (:selected-api db) :input-value])))

(reg-sub
 ::last-response
 (fn [db _]
   (get-in db [:apis-data (:selected-api db) :last-response])))

(reg-sub
 ::response-message
 :<- [::last-response]
 (fn [last-response]
   (.stringify js/JSON (.parse js/JSON last-response) (clj->js nil) (clj->js 2))))
