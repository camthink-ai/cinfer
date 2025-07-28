import { createSvgIcon } from '@mui/material/utils';

const FileUnknownIcon = createSvgIcon(
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="currentColor"
    >
        <path
            d="M0 6C0 2.68629 2.68629 0 6 0H18C21.3137 0 24 2.68629 24 6V18C24 21.3137 21.3137 24 18 24H6C2.68629 24 0 21.3137 0 18V6Z"
            fill="currentColor"
        />
        <path
            fillRule="evenodd"
            clipRule="evenodd"
            d="M6.72139 7.09091C6.02172 7.09091 5.45453 7.67701 5.45453 8.4V15.8182H18.5454V10.6372C18.5454 9.91425 17.9782 9.32815 17.2786 9.32815H11.4731L9.81042 7.09091H6.72139ZM18.5454 16.6909V16.2545H5.45453V16.6909C5.45453 17.4139 6.02172 18 6.72139 18H17.2786C17.9782 18 18.5454 17.4139 18.5454 16.6909Z"
            fill="white"
        />
        <path
            opacity="0.7"
            d="M10.9091 7.63637H17.4545V8.72728H11.8503L10.9091 7.63637Z"
            fill="white"
        />
    </svg>,
    'FileUnknownIcon',
);

export default FileUnknownIcon;
