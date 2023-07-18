(ns openllm.components.subs
  (:require [openllm.subs :as root-subs]
            [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::chat-db
 :<- [::root-subs/components-db]
 (fn [components-db _]
   (:chat-db components-db)))
