import Input from "components/Common/Input";
import { MouseEvent } from "react";
import SubmitButton from "components/Auth/SubmitButton";
import { useAppDispatch, useAppSelector } from "app/store/Hooks";
import { signIn } from "app/store/ducks/auth/authThunk";
import useInput from "hooks/useInput";
import Loading from "components/Common/Loading";
import { authAction } from 'app/store/ducks/auth/authSlice';

const placeholder = {
    username: "username",
    password: "password",
};

export default function LoginFormAndButton() {
    const [usernameInputProps, usernameIsValid, usernameIsFocus] = useInput(
        "",
        undefined,
        (value) => value.length > 0,
    );
    const [passwordInputProps, passwordIsValid, passwordIsFocus] = useInput(
        "",
        undefined,
        (value) => value.length > 5,
    );
    const dispatch = useAppDispatch();
    const isLoading = useAppSelector((state) => state.auth.isLoading);

    const submitButtonClickHandler = (event: MouseEvent<HTMLButtonElement>) => {
        event.preventDefault();
        // const requestSignIn = async () => {
        //     await dispatch(
        //         signIn({
        //             username: usernameInputProps.value,
        //             password: passwordInputProps.value,
        //         }),
        //     );
        // };
        dispatch(authAction.login());
    };

    return (
        <>
            <Input
                type="text"
                inputName="username"
                innerText={placeholder.username}
                inputProps={usernameInputProps}
                isFocus={usernameIsFocus}
            />
            <Input
                type="password"
                inputName="password"
                innerText={placeholder.password}
                inputProps={passwordInputProps}
                isFocus={passwordIsFocus}
            />
            <SubmitButton
                type="submit"
                onClick={submitButtonClickHandler}
                disabled={!(usernameIsValid && passwordIsValid)}
            >
                {isLoading ? <Loading size={18} isInButton={true} /> : "Login"}
            </SubmitButton>
        </>
    );
}
