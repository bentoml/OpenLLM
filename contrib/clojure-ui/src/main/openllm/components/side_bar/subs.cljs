(ns openllm.components.side-bar.subs
  (:require [openllm.components.subs :as components-subs]
            [reagent-mui.icons.keyboard-double-arrow-right :as right-icon]
            [reagent-mui.icons.keyboard-double-arrow-left :as left-icon]
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

(reg-sub
 ::tooltip-text-collapse-sidebar
 :<- [::side-bar-open?]
 (fn [side-bar-open? _]
   (str (if side-bar-open? "Collapse" "Expand") " side bar")))

(reg-sub
 ::collapse-icon
 :<- [::side-bar-open?]
 (fn [side-bar-open? _]
   (if side-bar-open?
     [right-icon/keyboard-double-arrow-right]
     [left-icon/keyboard-double-arrow-left])))
