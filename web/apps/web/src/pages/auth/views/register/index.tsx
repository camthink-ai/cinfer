import { useNavigate } from 'react-router-dom';
import { useForm, Controller, type SubmitHandler } from 'react-hook-form';
import { Paper, Typography, Button, Box } from '@mui/material';
import { useI18n } from '@milesight/shared/src/hooks';
import { toast } from '@milesight/shared/src/components';
import {
    iotLocalStorage,
    TOKEN_CACHE_KEY,
    REGISTERED_KEY,
} from '@milesight/shared/src/utils/storage';
import { globalAPI, awaitWrap, isRequestSuccess } from '@/services/http';
import useFormItems, { type FormDataProps } from '../useFormItems';
import './style.less';

export default () => {
    const navigate = useNavigate();

    // ---------- Form data processing ----------
    const { getIntlText } = useI18n();
    const { handleSubmit, control, watch } = useForm<FormDataProps>();
    const formItems = useFormItems({ mode: 'register', latestPassword: watch('password') });

    const onSubmit: SubmitHandler<FormDataProps> = async data => {
        const { username, password } = data;
        const [error, resp] = await awaitWrap(
            globalAPI.oauthRegister({
                email: '',
                nickname: username!,
                password,
            }),
        );

        if (error || !isRequestSuccess(resp)) return;

        navigate('/auth/login');
        iotLocalStorage.setItem(REGISTERED_KEY, true);
        // Clear existing TOKEN data to prevent new users from logging in
        iotLocalStorage.removeItem(TOKEN_CACHE_KEY);
        toast.success(getIntlText('auth.message.register_success'));
    };

    return (
        <Box
            component="form"
            onSubmit={handleSubmit(onSubmit)}
            className="ms-view-register ms-gradient-background"
        >
            <Paper className="ms-auth-container" elevation={3}>
                <div className="ms-auth-logo">{getIntlText('common.label.account_initialize')}</div>
                <Typography align="center" variant="body2" color="textSecondary">
                    {getIntlText('common.message.register_helper_text')}
                </Typography>
                <div className="ms-auth-form">
                    {formItems.map(props => (
                        <Controller<FormDataProps> key={props.name} {...props} control={control} />
                    ))}
                </div>
                <Button
                    fullWidth
                    type="submit"
                    sx={{ mt: 2.5, textTransform: 'none' }}
                    variant="contained"
                    className="ms-auth-submit"
                >
                    {getIntlText('common.button.confirm')}
                </Button>
            </Paper>
        </Box>
    );
};
