(ns openllm.components.subs
  "The subscriptions from this namespace are used to access the components'
   db branches.
   Every subscription in this namespace is an extractor (layer 2) of the
   `app-db`."
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

(reg-sub
 ::nav-bar-db
 :<- [::root-subs/components-db]
 (fn [components-db _]
   (:nav-bar-db components-db)))

(reg-sub
  ::playground-db
  :<- [::root-subs/components-db]
  (fn [components-db _]
    (:playground-db components-db)))

(reg-sub
  ::side-bar-db
  :<- [::root-subs/components-db]
  (fn [components-db _]
    (:side-bar-db components-db)))
