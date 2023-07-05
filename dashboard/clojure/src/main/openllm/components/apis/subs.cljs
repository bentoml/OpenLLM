(ns openllm.components.apis.subs
  (:require [clojure.pprint :as pprint]
            [clojure.string :as str]
            [re-frame.core :refer [reg-sub]]))

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
   (if (map? last-response)
     (with-out-str (pprint/pprint last-response))
     (try (.stringify js/JSON (.parse js/JSON last-response) (clj->js nil) (clj->js 2))
          (catch js/Error _
            (str last-response))))))
