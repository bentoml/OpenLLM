(ns openllm.components.side-bar.model-selection.subs
    (:require [openllm.components.side-bar.model-selection.db :as db]
              [openllm.components.side-bar.subs :as side-bar-subs]
              [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::selected-model
 :<- [::side-bar-subs/model-selection-db]
 (fn [model-selection-db _]
   (:selected-model model-selection-db)))

(reg-sub
 ::all-models-data
 :<- [::side-bar-subs/model-selection-db]
 (fn [model-selection-db _]
   (:all-models model-selection-db)))

(reg-sub
 ::selected-model-type
 :<- [::selected-model]
 (fn [selected-model _]
   (:model-type selected-model)))

(reg-sub
 ::selected-model-id
 :<- [::selected-model]
 (fn [selected-model _]
   (:model-id selected-model)))

;; Returns a list of all `model-types`.
(reg-sub
 ::all-model-types
 :<- [::all-models-data]
 (fn [all-models-data _]
   (keys all-models-data)))

;; Returns a list of all `model-ids` for the current `model-type`.
(reg-sub
 ::all-model-ids
 :<- [::selected-model-type]
 :<- [::all-models-data]
 (fn [[selected-model-type all-models-data] _]
   (get-in all-models-data
           [selected-model-type :model_id]
           [db/loading-text])))
