(ns openllm.components.model-selection.subs
    (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::selected-model-type
 :<- [:selected-model]
 (fn [selected-model _]
   (:model-type selected-model)))

(reg-sub
 ::selected-model-id
 :<- [:selected-model]
 (fn [selected-model _]
   (:model-id selected-model)))
