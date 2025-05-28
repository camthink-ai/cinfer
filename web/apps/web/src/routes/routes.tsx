import intl from 'react-intl-universal';
import { RouteObject } from 'react-router-dom';

import {
    HomeOutlinedIcon,
    SmartToyOutlinedIcon,
    LockOutlinedIcon,
} from '@milesight/shared/src/components';

import ErrorBoundaryComponent from './error-boundary';

type RouteObjectType = RouteObject & {
    /** Custom routing metadata */
    handle?: {
        title?: string;

        /** Menu icon */
        icon?: React.ReactNode;

        /**
         * Layout type, default is' basic '
         *
         * Note: The type here should be LayoutType, but inference errors can occur, so it is temporarily defined as string
         */
        layout?: string;

        /** Whether to access without login, default 'false' (login required) */
        authFree?: boolean;

        /**
         * Whether to hide in the menu bar
         */
        hideInMenuBar?: boolean;

        /** Hide the sidebar */
        hideSidebar?: boolean;
    };

    /** subroute */
    children?: RouteObjectType[];
};

const ErrorBoundary = () => <ErrorBoundaryComponent inline />;

const routes: RouteObjectType[] = [
    {
        path: '/dashboard',
        handle: {
            get title() {
                return intl.get('common.label.dashboard');
            },
            icon: <HomeOutlinedIcon fontSize="small" />,
        },
        async lazy() {
            const { default: Component } = await import('@/pages/dashboard');
            return { Component };
        },
        ErrorBoundary,
    },
    {
        path: '/model',
        handle: {
            get title() {
                return intl.get('common.label.model_management');
            },
            icon: <SmartToyOutlinedIcon fontSize="small" />,
        },
        async lazy() {
            const { default: Component } = await import('@/pages/model');
            return { Component };
        },
        ErrorBoundary,
    },
    {
        path: '/token',
        handle: {
            get title() {
                return intl.get('common.label.token_management');
            },
            icon: <LockOutlinedIcon fontSize="small" />,
        },
        async lazy() {
            const { default: Component } = await import('@/pages/token');
            return { Component };
        },
        ErrorBoundary,
    },
    {
        path: '/auth',
        handle: {
            layout: 'blank',
        },
        async lazy() {
            const { default: Component } = await import('@/pages/auth');
            return { Component };
        },
        ErrorBoundary,
        children: [
            {
                index: true,
                path: 'login',
                handle: {
                    get title() {
                        return intl.get('common.label.login');
                    },
                    layout: 'blank',
                },
                async lazy() {
                    const { default: Component } = await import('@/pages/auth/views/login');
                    return { Component };
                },
                ErrorBoundary,
            },
            {
                path: 'register',
                handle: {
                    get title() {
                        return intl.get('common.label.register');
                    },
                    layout: 'blank',
                },
                async lazy() {
                    const { default: Component } = await import('@/pages/auth/views/register');
                    return { Component };
                },
                ErrorBoundary,
            },
        ],
    },
    {
        path: '/403',
        handle: {
            title: '403',
            hideInMenuBar: true,
        },
        async lazy() {
            const { default: Component } = await import('@/pages/403');
            return { Component };
        },
        ErrorBoundary,
    },
    {
        path: '*',
        handle: {
            title: '404',
            layout: 'blank',
            authFree: true,
        },
        async lazy() {
            const { default: Component } = await import('@/pages/404');
            return { Component };
        },
        ErrorBoundary,
    },
];

export default routes;
