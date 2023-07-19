(ns openllm.components.playground.subs
  (:require [openllm.components.subs :as components-subs]
            [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::playground-input-value
 :<- [::components-subs/playground-db]
 (fn [playground-db _]
   (:playground-input-value playground-db)))

(reg-sub
 ::last-response
 :<- [::components-subs/playground-db]
 (fn [playground-db _]
   (:playground-last-response playground-db)))

(reg-sub
 ::response-modal-open?
 :<- [::components-subs/playground-db]
 (fn [playground-db _]
   (:response-modal-open? playground-db)))
