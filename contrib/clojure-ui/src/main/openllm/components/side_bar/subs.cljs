(ns openllm.components.side-bar.subs
  (:require [openllm.components.subs :as components-subs]
            [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::side-bar-open?
 :<- [::components-subs/side-bar-db]
 (fn [side-bar-db _]
   (:side-bar-open? side-bar-db)))

(reg-sub
 ::model-selection-db
 :<- [::components-subs/side-bar-db]
 (fn [side-bar-db _]
   (:model-selection-db side-bar-db)))

(reg-sub
 ::model-params-db
 :<- [::components-subs/side-bar-db]
 (fn [side-bar-db _]
   (:model-params-db side-bar-db)))
