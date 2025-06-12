import { useMemoizedFn } from 'ahooks';

import { useI18n } from '@milesight/shared/src/hooks';
import { toast } from '@milesight/shared/src/components';

import { useConfirm } from '@/components';
import { type TokenItemProps, tokenAPI, awaitWrap, isRequestSuccess } from '@/services/http';

export default function useToken(getAllTokens?: () => void) {
    const { getIntlText } = useI18n();
    const confirm = useConfirm();

    const changeTokenEnableStatus = useMemoizedFn(
        async (record: TokenItemProps, checked: boolean) => {
            if (!record?.id) return;

            const [err, resp] = await awaitWrap(
                checked
                    ? tokenAPI.enableToken({
                          access_token_id: record.id,
                      })
                    : tokenAPI.disableToken({ access_token_id: record.id }),
            );
            if (err || !isRequestSuccess(resp)) {
                return;
            }

            toast.success(getIntlText('common.message.operation_success'));
            getAllTokens?.();
        },
    );

    const handleDeleteToken = useMemoizedFn((record: TokenItemProps) => {
        confirm({
            title: getIntlText('common.label.delete'),
            description: getIntlText('token.tip.delete_token'),
            confirmButtonText: getIntlText('common.label.delete'),
            confirmButtonProps: {
                color: 'error',
            },
            onConfirm: async () => {
                if (!record?.id) return;

                const [err, resp] = await awaitWrap(
                    tokenAPI.deleteToken({
                        access_token_id: record.id,
                    }),
                );

                if (err || !isRequestSuccess(resp)) {
                    return;
                }

                getAllTokens?.();
                toast.success(getIntlText('common.message.delete_success'));
            },
        });
    });

    return {
        /**
         * enable/disable the token
         */
        changeTokenEnableStatus,
        /**
         * to delete token
         */
        handleDeleteToken,
    };
}
