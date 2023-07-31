(ns openllm.components.side-bar.views
  (:require [re-frame.core :as rf]
            [openllm.components.side-bar.model-selection.views :as model-selection-views]
            [openllm.components.side-bar.model-params.views :as model-params-views]
            [openllm.components.side-bar.subs :as subs]
            [openllm.components.side-bar.events :as events]
            [openllm.components.common.views :as ui]
            [reagent-mui.material.button :refer [button]]
            [reagent-mui.material.tooltip :refer [tooltip]]
            [reagent-mui.material.collapse :refer [collapse]]))

(defn- collapse-side-bar-button
  "The collapse side bar button. It changes its icon depending on whether
   the side bar is collapsed or not."
  []
  (let [tooltip-text-collapse-sidebar (rf/subscribe [::subs/tooltip-text-collapse-sidebar])
        collapse-icon (rf/subscribe [::subs/collapse-icon])]
    (fn []
      [:div {:class "-ml-4"}
       [tooltip {:title @tooltip-text-collapse-sidebar}
        [button {:on-click #(rf/dispatch [::events/toggle-side-bar])
                 :color "secondary"
                 :class "h-4"
                 :style {:max-width "16px"
                         :min-width "16px"
                         :background-color "black"
                         :border-radius "8px 0px 0px 8px"
                         :margin-top "-2px"}}
         [:div {:class "-mr-1 -mt-px"}
          @collapse-icon]]]])))

(defn side-bar-with-mui-collapse
  "The sidebar wrapped with a Material UI Collapse component. The collapse
   component is used to animate the sidebar when it is opened or closed."
  []
  (let [side-bar-open? (rf/subscribe [::subs/side-bar-open?])]
    (fn []
      [collapse {:in @side-bar-open?
                 :orientation "horizontal"
                 :class "flex flex-col w-80"}
       [:div {:class "flex flex-col w-80 h-full border-l border-gray-200 pt-0.5 pb-4"}
        [model-selection-views/model-selection]
        [:hr {:class "mb-2 border-1 border-gray-200"}]
        [ui/headline "Parameters" 2]
        [:div {:class "my-0 h-0 flex-1 flex flex-col overflow-y-auto scrollbar"}
         [:div {:class "px-3 mt-0 relative inline-block text-left"}
          [model-params-views/parameter-list]]]]])))

(defn side-bar
  "The render function of the toolbar on the very right of the screen. Contains the
   model selection dropdowns and the parameter list."
  []
  [:<>
   [collapse-side-bar-button]
   [:div {:class "hidden lg:flex lg:flex-shrink-0"}
    [side-bar-with-mui-collapse]]])
