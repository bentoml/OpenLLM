(ns openllm.components.subs
  (:require [openllm.subs :as root-subs]
            [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-db
 :<- [::root-subs/components-db]
 (fn [components-db _]
   (:chat-db components-db)))

(reg-sub
 ::model-selection-db
 :<- [::root-subs/components-db]
 (fn [components-db _]
   (:model-selection-db components-db)))
