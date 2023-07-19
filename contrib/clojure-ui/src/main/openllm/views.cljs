(ns openllm.views
  "This is the root views namespace, while the first DOM/hiccup is created in
   the `openllm.app` namespace, this namespace is the first to be pure and
   only create hiccup/DOM, derived from data :)
   The `openllm.app` namespace dealt with the initialisation of the `app-db`,
   created the root DOM/hiccup node and handeled the material-ui theming.
   
   From this point onward all the views are pure and only depend on the `app-db`,
   which is queried by subscriptions and the single source of truth for the
   entire application."
  (:require [re-frame.core :as rf]
            [openllm.components.nav-bar.views :as nav-bar-views]
            [openllm.components.side-bar.views :as side-bar-views]
            [openllm.components.playground.views :as playground-views]
            [openllm.components.chat.views :as chat-views]))

(defn tab-content
  "The content of the currently active tab. Essentially, this is the
   main router of the app, deciding what content should be rendered
   based on the current screen-id."
  [screen-id]
  (case screen-id
    :playground [playground-views/playground-tab-contents]
    :chat [chat-views/chat-tab-contents]))

(defn dashboard
  "The main dashboard component, which is rendered into the DOM. Called
   directly by the `app` function residing inside the `app` namespace,
   which is the main component and rendered by the root rendering function
   of this application. Called is putting it mildly, as this function is
   basically wrapped by the `app` function, thus this component contains
   everything that is rendered into the DOM as it's children."
  []
  (let [screen-id (rf/subscribe [:screen-id])]
    (fn []
      [:div {:class "h-screen flex bg-white"}
       [:div {:class "flex flex-col flex-1 w-screen"}
        [:main {:class "flex-1 relative z-0 overflow-hidden focus:outline-none" :tabIndex "0"}
         [nav-bar-views/nav-bar]
         [:div {:class "mt-2 pr-0.5 pl-4 w-full h-full"}
          [tab-content @screen-id]]]]
       [side-bar-views/side-bar]])))
