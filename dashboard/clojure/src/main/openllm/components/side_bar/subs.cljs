(ns openllm.components.side-bar.subs
  (:require [re-frame.core :refer [reg-sub]]))

(reg-sub
 ::side-bar-open?
 (fn [db _]
   (:side-bar-open? db)))
